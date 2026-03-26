#!/usr/bin/env python3
"""
score_spectrogram.py

Full pipeline — no subprocess calls:
  1. Download audio from a YouTube URL via the yt-dlp Python API
  2. Separate stems via the demucs Python API (htdemucs_6s model)
  3. Render a 4-staff CQT spectrogram score image

Staff order (top → bottom, like a grand-staff score):
  ┌─────────────────────────────────┐
  │  Piano   — full piano range (A0–C8), pitch-annotated in scientific notation
  ├─────────────────────────────────┤
  │  Other   — vocals + guitar + other stems combined, pitch-annotated
  ├─────────────────────────────────┤
  │  Bass    — pitch-annotated
  ├─────────────────────────────────┤
  │  Drums   — zoomed to drums energy band (default 30 Hz–12 kHz),
  │            y-axis shows frequency in Hz (log), no pitch-name annotations;
  │            landmark labels: Sub, Kick, Snare, Toms, Shell, HH, Cym
  └─────────────────────────────────┘

All pitched staves share the same CQT pitch grid (C4 = middle C highlighted white).
The drums staff uses its own independent CQT window.

Dependencies:
    (Install demucs from Github to resolve packaging issue.)
    /Users/trqkdata/Projects/env/piano_transcribe/bin/pip install \
  "git+https://github.com/facebookresearch/demucs#egg=demucs"
    (pip install "llvmlite==0.41.1" --prefer-binary) (Intel Macs)
    (pip install "numba==0.58.1" --prefer-binary) (Intel Macs)
    pip install yt-dlp demucs librosa matplotlib numpy torch

Usage:
    python score_spectrogram.py <youtube_url> [options]
    python score_spectrogram.py --file <path> [options]

Examples:
    # Download from YouTube (full track)
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE"

    # Download from YouTube, first 60 seconds only
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" --duration 60

    # Download from YouTube, seconds 30–90
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" --start 30 --end 90

    # Use a local file (full track)
    python score_spectrogram.py --file ./work/audio/song.wav

    # Use a local file, first 60 seconds only
    python score_spectrogram.py --file ./work/audio/song.wav --duration 60

    # Use a local file, seconds 30–90
    python score_spectrogram.py --file ./work/audio/song.wav --start 30 --end 90

    # Custom output and display options
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" \\
        --px-per-sec 60 --staff-height 280 --out my_score.png
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" \\
        --drums-fmin 20 --drums-fmax 16000
"""

import argparse
import itertools
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Per-staff CQT configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StaffConfig:
    name: str
    cmap: str
    annotate_pitch: bool    # draw C-note lines and note-name labels?
    fmin_hz: float          # CQT lower bound in Hz
    fmax_hz: float          # CQT upper bound in Hz
    bins_per_octave: int    # frequency resolution
    max_pitches: int = 6    # max fundamental pitch labels per onset

    @property
    def n_bins(self) -> int:
        return int(np.ceil(self.bins_per_octave * np.log2(self.fmax_hz / self.fmin_hz)))


# Landmark frequency annotations for the drums staff (Hz, label)
DRUM_LANDMARKS = [
    (30,    "Sub"),
    (60,    "Kick"),
    (200,   "Snare"),
    (500,   "Toms"),
    (2000,  "Shell"),
    (5000,  "HH"),
    (10000, "Cym"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Download
# ─────────────────────────────────────────────────────────────────────────────

def download_audio(url: str, output_dir: Path) -> Path:
    """
    Download audio from a YouTube URL as a WAV using the yt-dlp Python API.
    Returns the path to the downloaded WAV file.
    """
    try:
        import yt_dlp
    except ImportError:
        sys.exit("yt-dlp not found.  Run: pip install yt-dlp")

    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base = ydl.prepare_filename(info)
        wav_path = Path(base).with_suffix(".wav")
        if not wav_path.exists():
            wav_path = Path(base.rsplit(".", 1)[0] + ".wav")

    if not wav_path.exists():
        sys.exit(f"Expected WAV at {wav_path} but file not found after download.")

    print(f"[download]  -> {wav_path}")
    return wav_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Stem separation
# ─────────────────────────────────────────────────────────────────────────────

_OTHER_MERGE = ("vocals", "guitar", "other")


def separate_stems(
    audio_path: Path,
    start_s: float | None = None,
    end_s: float | None = None,
) -> tuple[dict[str, np.ndarray], int, np.ndarray]:
    """
    Two-pass demucs separation with shifts=10 for quality.

    Pass 1 — htdemucs (4-stem): yields final Drums and Bass stems.
    Pass 2 — htdemucs_6s (6-stem) on pass-1 "other": yields Piano and Other.

    If start_s/end_s are given the audio is trimmed BEFORE separation so
    demucs only processes the requested segment.

    Returns:
        stems  -- dict with keys 'Piano', 'Other', 'Bass', 'Drums',
                  each a mono float32 numpy array
        sr     -- sample rate (int)
        mixed  -- mono float32 numpy array of the (trimmed) input mix
    """
    try:
        from demucs.api import Separator
    except ImportError:
        sys.exit("demucs not found.  Run: pip install demucs")
    try:
        import torchaudio
    except ImportError:
        sys.exit("torchaudio not found.  Run: pip install torchaudio")

    def to_mono(tensor: torch.Tensor) -> np.ndarray:
        return tensor.mean(dim=0).cpu().numpy().astype(np.float32)

    # ── Load & trim ───────────────────────────────────────────────────────
    print("[demucs]  Loading model htdemucs (4-stem, pass 1) ...")
    sep4 = Separator(model="htdemucs", shifts=10)
    sr = sep4.samplerate  # 44100

    wav, file_sr = torchaudio.load(str(audio_path))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    if start_s is not None:
        wav = wav[:, int(start_s * sr):int(end_s * sr)]
        print(f"[demucs]  Pass 1: {audio_path.name} [{start_s:.1f}s – {end_s:.1f}s] ...")
    else:
        print(f"[demucs]  Pass 1: {audio_path.name} ...")

    mixed = to_mono(wav)

    # ── Pass 1: htdemucs → Drums, Bass, and the residual "other" ─────────
    done = threading.Event()
    t = threading.Thread(target=_spin, args=("htdemucs pass 1", done), daemon=True)
    t.start()
    _, raw4 = sep4.separate_tensor(wav)
    done.set(); t.join()
    del sep4  # free GPU/CPU memory before loading second model

    # ── Pass 2: htdemucs_6s on pass-1 "other" → Piano, Other ─────────────
    print("[demucs]  Loading model htdemucs_6s (6-stem, pass 2) ...")
    sep6 = Separator(model="htdemucs_6s", shifts=10)
    if sep6.samplerate != sr:
        other_in = torchaudio.functional.resample(raw4["other"], sr, sep6.samplerate)
    else:
        other_in = raw4["other"]
    print("[demucs]  Pass 2: extracting piano / other from 4-stem residual ...")
    done = threading.Event()
    t = threading.Thread(target=_spin, args=("htdemucs_6s pass 2", done), daemon=True)
    t.start()
    _, raw6 = sep6.separate_tensor(other_in)
    done.set(); t.join()
    del sep6

    other6_combined = sum(raw6[k] for k in _OTHER_MERGE if k in raw6)

    stems = {
        "Piano": to_mono(raw6["piano"]),
        "Other": to_mono(other6_combined),
        "Bass":  to_mono(raw4["bass"]),
        "Drums": to_mono(raw4["drums"]),
    }

    for name, arr in stems.items():
        print(f"[demucs]  {name:8s}: {arr.shape[0] / sr:.1f}s")

    return stems, sr, mixed


# ─────────────────────────────────────────────────────────────────────────────
# Audio export
# ─────────────────────────────────────────────────────────────────────────────

def save_audio_outputs(
    mixed: np.ndarray,
    stems: dict[str, np.ndarray],
    sr: int,
    output_dir: Path,
    stem_name: str,
):
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("soundfile not found.  Run: pip install soundfile")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"mixed": mixed, **{k.lower(): v for k, v in stems.items()}}
    for label, audio in files.items():
        path = output_dir / f"{stem_name}_{label}.wav"
        sf.write(str(path), audio, sr)
        print(f"[audio]   Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: CQT computation
# ─────────────────────────────────────────────────────────────────────────────

def _spin(label: str, done: threading.Event):
    start = time.time()
    for ch in itertools.cycle(r'\|/-'):
        if done.is_set():
            break
        elapsed = time.time() - start
        print(f"\r  {ch} {label} ... {elapsed:.0f}s", end="", flush=True)
        time.sleep(0.2)
    print()


def compute_cqt_db(
    audio: np.ndarray,
    sr: int,
    hop_length: int,
    cfg: StaffConfig,
) -> np.ndarray:
    C = librosa.cqt(
        audio,
        sr=sr,
        hop_length=hop_length,
        fmin=cfg.fmin_hz,
        n_bins=cfg.n_bins,
        bins_per_octave=cfg.bins_per_octave,
        res_type="kaiser_best",
    )
    return librosa.amplitude_to_db(np.abs(C), ref=np.max)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Onset detection + pitch annotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pitches_at_onset(
    cqt_db: np.ndarray,
    frame: int,
    next_frame: int,
    cfg: StaffConfig,
    top_n: int = 6,
    threshold_db: float = -28,
) -> list[tuple[float, str]]:
    """
    Return (hz, note_name) pairs for prominent fundamentals in the onset window.

    Takes the per-bin maximum over [frame, next_frame), finds peaks above a
    relative threshold, and returns the top_n by energy sorted low→high.
    """
    from scipy.signal import find_peaks

    end = min(max(frame + 1, next_frame), cqt_db.shape[1])
    spectrum = cqt_db[:, frame:end].max(axis=1)
    window_max = spectrum.max()
    if window_max < -80:   # silence
        return []

    peaks, _ = find_peaks(
        spectrum,
        height=window_max + threshold_db,          # relative to window peak
        distance=max(1, cfg.bins_per_octave // 6), # ≥ 2 semitones apart
    )
    if not len(peaks):
        return []

    # Keep top_n by energy, then re-sort by frequency for display
    order = np.argsort(spectrum[peaks])[::-1][:top_n]
    peaks = np.sort(peaks[order])

    return [
        (cfg.fmin_hz * 2.0 ** (p / cfg.bins_per_octave),
         librosa.hz_to_note(cfg.fmin_hz * 2.0 ** (p / cfg.bins_per_octave)))
        for p in peaks
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Render
# ─────────────────────────────────────────────────────────────────────────────

BG         = "#111111"
GRID_DIM   = "#555555"
GRID_C4    = "#ffffff"
LABEL_DIM  = "#888888"
LABEL_C4   = "#ffffff"
DRUM_COLOR = "#aaaaaa"


def _draw_pitch_annotations(ax, duration: float, cfg: StaffConfig):
    """Draw horizontal C-note lines + note-name labels across a pitched staff."""
    for octave in range(0, 9):
        hz = librosa.note_to_hz(f"C{octave}")
        if not (cfg.fmin_hz <= hz <= cfg.fmax_hz):
            continue
        is_c4 = (octave == 4)
        ax.axhline(
            y=hz,
            color=GRID_C4 if is_c4 else GRID_DIM,
            linewidth=1.2 if is_c4 else 0.4,
            alpha=0.85, zorder=2,
        )
        ax.text(
            duration * 0.001, hz,
            f"C{octave}",
            color=LABEL_C4 if is_c4 else LABEL_DIM,
            fontsize=6, va="bottom", zorder=3,
            transform=ax.transData,
        )


def _draw_drum_annotations(ax, duration: float, cfg: StaffConfig):
    """
    Draw landmark frequency lines + labels on the drums staff.
    No pitch-name annotations -- labels describe instrument register instead.
    """
    for hz, label in DRUM_LANDMARKS:
        if not (cfg.fmin_hz <= hz <= cfg.fmax_hz):
            continue
        ax.axhline(y=hz, color=DRUM_COLOR, linewidth=0.4, alpha=0.6, zorder=2)
        ax.text(
            duration * 0.001, hz, label,
            color=DRUM_COLOR, fontsize=6, va="bottom", zorder=3,
            transform=ax.transData,
        )


def render_score(
    stems: dict[str, np.ndarray],
    sr: int,
    output_path: Path,
    staff_configs: list[StaffConfig],
    px_per_sec: int = 60,
    staff_height_px: int = 280,
    dpi: int = 100,
):
    hop_length = max(1, int(sr / px_per_sec))
    duration = len(stems["Piano"]) / sr
    width_px = int(duration * px_per_sec)
    total_height_px = staff_height_px * len(staff_configs)

    print(f"[render]  Image: {width_px} x {total_height_px} px  "
          f"({duration:.1f}s @ {px_per_sec} px/s)")

    # Pre-compute all CQTs
    cqts = {}
    for cfg in staff_configs:
        fmin_note = librosa.hz_to_note(cfg.fmin_hz)
        fmax_note = librosa.hz_to_note(cfg.fmax_hz)
        print(f"[render]  CQT for {cfg.name} "
              f"({fmin_note}-{fmax_note}, {cfg.n_bins} bins) ...")
        done = threading.Event()
        t = threading.Thread(target=_spin, args=(f"CQT {cfg.name}", done), daemon=True)
        t.start()
        cqts[cfg.name] = compute_cqt_db(stems[cfg.name], sr, hop_length, cfg)
        done.set()
        t.join()

    # Onset detection per stem
    print("[render]  Detecting onsets ...")
    onset_data = {}  # cfg.name -> (frames_array, times_array)
    for cfg in staff_configs:
        frames = librosa.onset.onset_detect(
            y=stems[cfg.name], sr=sr, hop_length=hop_length
        )
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
        onset_data[cfg.name] = (frames, times)
        print(f"[render]  {cfg.name:8s}: {len(frames)} onsets")

    # ── Figure layout ─────────────────────────────────────────────────────
    _REF_DPI = 100  # figsize is defined at this density; dpi scales output pixels
    fig = plt.figure(
        figsize=(width_px / _REF_DPI, total_height_px / _REF_DPI),
        dpi=dpi, facecolor=BG,
    )
    gs = gridspec.GridSpec(
        len(staff_configs), 1,
        figure=fig,
        hspace=0.0,
        top=0.97, bottom=0.04,
        left=0.055, right=0.985,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(len(staff_configs))]

    for row, cfg in enumerate(staff_configs):
        ax = axes[row]
        ax.set_facecolor(BG)
        is_bottom = (row == len(staff_configs) - 1)

        # Pitched staves: 'cqt_note' puts scientific pitch names on y-axis.
        # Drums staff: 'cqt_hz' shows frequency in Hz on a log scale --
        # honest about what we're looking at, avoids misleading pitch names.
        y_axis_type = "cqt_note" if cfg.annotate_pitch else "cqt_hz"

        librosa.display.specshow(
            cqts[cfg.name],
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis=y_axis_type,
            fmin=cfg.fmin_hz,
            bins_per_octave=cfg.bins_per_octave,
            ax=ax,
            cmap=cfg.cmap,
        )

        # Overlay annotations
        if cfg.annotate_pitch:
            _draw_pitch_annotations(ax, duration, cfg)
            ax.set_yticklabels([])   # suppress librosa default; we draw our own
        else:
            _draw_drum_annotations(ax, duration, cfg)
            ax.tick_params(axis="y", colors=DRUM_COLOR, labelsize=6)

        # Onset lines (all staves)
        # All specshow calls now use x_axis="time", so x coordinates are seconds.
        frames_arr, times_arr = onset_data[cfg.name]
        for t in times_arr:
            ax.axvline(x=t, color="red", linewidth=0.4, alpha=0.5, zorder=4)

        # Pitch labels at each onset (pitched staves only)
        if cfg.annotate_pitch:
            n_frames = cqts[cfg.name].shape[1]
            for i, (frame, t) in enumerate(zip(frames_arr, times_arr)):
                next_frame = int(frames_arr[i + 1]) if i + 1 < len(frames_arr) else n_frames
                for hz, note in _pitches_at_onset(cqts[cfg.name], int(frame), next_frame, cfg, top_n=cfg.max_pitches):
                    ax.text(
                        t, hz, note,
                        color="red", fontsize=4, va="bottom", alpha=0.85,
                        zorder=5, clip_on=True, transform=ax.transData,
                    )

        # Staff label on left margin
        ax.set_ylabel(cfg.name, color="#cccccc", fontsize=8, rotation=90, labelpad=4)
        ax.yaxis.set_label_position("left")

        # Tick styling
        ax.tick_params(axis="both", colors="#888888", labelsize=6, length=2, pad=2)

        if not is_bottom:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.xaxis.label.set_color("#aaaaaa")
            ax.tick_params(axis="x", colors="#aaaaaa", labelsize=7)

        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
            spine.set_linewidth(0.5)

    # ── Save ──────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        n = 1
        while True:
            candidate = output_path.with_stem(f"{output_path.stem}_{n}")
            if not candidate.exists():
                output_path = candidate
                break
            n += 1
        print(f"[render]  Output exists; writing to {output_path.name}")
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[render]  Saved -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YouTube -> demucs -> 4-staff CQT score image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", nargs="?", default=None,
                        help="YouTube URL to download (omit if --file is given)")
    parser.add_argument("--file", "-f", default=None,
                        help="Path to an audio file already on disk (skips download)")
    parser.add_argument("--start", type=float, default=None,
                        help="Start offset in seconds for --file (use with --end)")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds for --file (use with --start)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Clip length in seconds from the start of --file")
    parser.add_argument("--out", "-o", default=None,
                        help="Output PNG path (default: ./output/<title>_score.png)")
    parser.add_argument("--work-dir", default="./work",
                        help="Directory for intermediate files (default: ./work)")
    parser.add_argument("--px-per-sec", type=int, default=60,
                        help="Horizontal pixels per second (default 60)")
    parser.add_argument("--staff-height", type=int, default=280,
                        help="Height in pixels per staff (default 280; total = 4x)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output DPI — scales pixel density above the 100-dpi "
                             "base layout (default 150)")

    # Pitched staves frequency range
    parser.add_argument("--fmin", default="A0",
                        help="Lowest pitch for Piano/Other/Bass staves (default A0)")
    parser.add_argument("--fmax", default="C8",
                        help="Highest pitch for Piano/Other/Bass staves (default C8)")
    parser.add_argument("--bins-per-octave", type=int, default=36,
                        help="CQT bins per octave for pitched staves (default 36)")

    # Drums staff frequency range — deliberately independent of pitched staves
    parser.add_argument("--drums-fmin", type=float, default=30.0,
                        help="Drums staff lower bound in Hz (default 30)")
    parser.add_argument("--drums-fmax", type=float, default=12000.0,
                        help="Drums staff upper bound in Hz (default 12000)")
    parser.add_argument("--drums-bins-per-octave", type=int, default=36,
                        help="CQT bins per octave for drums staff (default 36)")

    args = parser.parse_args()

    # Validate source / segment args
    if args.file and args.url:
        parser.error("--file and url are mutually exclusive")
    if not args.file and not args.url:
        parser.error("provide a YouTube url or --file PATH")
    if args.duration is not None and (args.start is not None or args.end is not None):
        parser.error("--duration and --start/--end are mutually exclusive")
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be used together")

    # Resolve pitched fmin/fmax from note names to Hz
    pitched_fmin = librosa.note_to_hz(args.fmin)
    pitched_fmax = librosa.note_to_hz(args.fmax)

    staff_configs = [
        StaffConfig("Piano", "magma",   annotate_pitch=True,
                    fmin_hz=pitched_fmin, fmax_hz=pitched_fmax,
                    bins_per_octave=args.bins_per_octave, max_pitches=10),
        StaffConfig("Other", "viridis", annotate_pitch=True,
                    fmin_hz=pitched_fmin, fmax_hz=pitched_fmax,
                    bins_per_octave=args.bins_per_octave, max_pitches=10),
        StaffConfig("Bass",  "cividis", annotate_pitch=True,
                    fmin_hz=pitched_fmin, fmax_hz=pitched_fmax,
                    bins_per_octave=args.bins_per_octave, max_pitches=1),
        StaffConfig("Drums", "inferno", annotate_pitch=False,
                    fmin_hz=args.drums_fmin, fmax_hz=args.drums_fmax,
                    bins_per_octave=args.drums_bins_per_octave),
    ]

    work_dir = Path(args.work_dir)

    # Stage 1: Download (or use existing file)
    if args.file:
        wav_path = Path(args.file)
        if not wav_path.exists():
            sys.exit(f"File not found: {wav_path}")
        print(f"[source]  Using {wav_path}")
    else:
        wav_path = download_audio(args.url, work_dir / "audio")

    # Stage 2: Separate
    if args.duration is not None:
        seg_start, seg_end = 0.0, args.duration
    elif args.start is not None:
        seg_start, seg_end = args.start, args.end
    else:
        seg_start, seg_end = None, None

    stems, sr, mixed = separate_stems(wav_path, start_s=seg_start, end_s=seg_end)

    # Stage 3: Save audio outputs
    out_dir = Path(args.out).parent if args.out else Path("./output")
    save_audio_outputs(mixed, stems, sr, out_dir, wav_path.stem)

    # Stage 4: Render
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("./output") / f"{wav_path.stem}_score.png"

    render_score(
        stems=stems,
        sr=sr,
        output_path=out_path,
        staff_configs=staff_configs,
        px_per_sec=args.px_per_sec,
        staff_height_px=args.staff_height,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
