#!/usr/bin/env python3
"""
score_spectrogram.py

Full pipeline — no subprocess calls:
  1. Download audio from a YouTube URL via the yt-dlp Python API
  2. Separate stems via the demucs Python API (htdemucs_6s model)
  3. Run pitch detection per pitched stem to get (onset, offset, midi_pitch) events
  4. Render a 4-staff CQT spectrogram score image with piano-roll note overlays

Staff order (top → bottom, like a grand-staff score):
  ┌─────────────────────────────────┐
  │  Piano   — piano_transcription_inference note overlay
  ├─────────────────────────────────┤
  │  Other   — basic-pitch note overlay
  ├─────────────────────────────────┤
  │  Bass    — basic-pitch note overlay
  ├─────────────────────────────────┤
  │  Drums   — zoomed 30 Hz–12 kHz, landmark labels, no pitch annotation
  └─────────────────────────────────┘

Note overlays draw a thin horizontal segment at the fundamental frequency
from onset to offset for each detected note — a piano roll burned into
the spectrogram.  Partials remain visible in the CQT energy but are not
annotated.  A single C4 reference line is the only persistent grid marker
on pitched staves.

Dependencies:
    # demucs must be installed from GitHub (PyPI 4.0.1 is missing api.py):
    pip install "git+https://github.com/facebookresearch/demucs#egg=demucs"
    # Intel Mac llvmlite fix:
    pip install "llvmlite==0.41.1" --prefer-binary
    pip install "numba==0.58.1" --prefer-binary
    # Main deps:
    pip install yt-dlp librosa matplotlib numpy torch torchaudio soundfile
    # Pitch detection (optional but recommended):
    pip install piano-transcription-inference
    pip install basic-pitch

Usage:
    python score_spectrogram.py <youtube_url> [options]
    python score_spectrogram.py --file <path> [options]

Examples:
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE"
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" --duration 60
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" --start 30 --end 90
    python score_spectrogram.py --file ./work/audio/song.wav
    python score_spectrogram.py --file ./work/audio/song.wav --start 30 --end 90
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" \\
        --note-confidence 0.6 --min-note-duration 0.05
    python score_spectrogram.py "https://youtu.be/ETzx-7LZdsE" \\
        --no-note-overlay   # pure CQT with C4 grid only, no detection
"""

import argparse
import itertools
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

class NoteEvent(NamedTuple):
    onset_s:    float   # note start in seconds
    offset_s:   float   # note end in seconds
    midi_pitch: int     # MIDI note number (60 = C4)
    confidence: float   # 0.0–1.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-staff CQT + rendering configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StaffConfig:
    name: str
    cmap: str
    annotate_pitch: bool    # True = pitched staff; False = drums
    fmin_hz: float
    fmax_hz: float
    bins_per_octave: int
    note_color: str         # color for note-overlay segments

    @property
    def n_bins(self) -> int:
        return int(np.ceil(self.bins_per_octave * np.log2(self.fmax_hz / self.fmin_hz)))


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
    shifts: int = 1,
) -> tuple[dict[str, np.ndarray], int]:
    try:
        from demucs.api import Separator
    except ImportError as e:
        sys.exit(f"Failed to import demucs: {e}")

    try:
        import torchaudio
    except ImportError:
        sys.exit("torchaudio not found.  Run: pip install torchaudio")

    def to_mono(tensor: torch.Tensor) -> np.ndarray:
        return tensor.mean(dim=0).cpu().numpy().astype(np.float32)

    # ── Load & trim ───────────────────────────────────────────────────────
    print("[demucs]  Loading htdemucs (4-stem: drums, bass, vocals, other) ...")
    sep4 = Separator(model="htdemucs", shifts=shifts, overlap=0.25)
    sr = sep4.samplerate  # 44100

    wav, file_sr = torchaudio.load(str(audio_path))
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    if start_s is not None:
        wav = wav[:, int(start_s * sr):int(end_s * sr)]
        print(f"[demucs]  Pass 1 (4-stem): [{start_s:.1f}s – {end_s:.1f}s], shifts={shifts} ...")
    else:
        print(f"[demucs]  Pass 1 (4-stem): {audio_path.name}, shifts={shifts} ...")

    # ── Pass 1: htdemucs → final Drums + Bass; "other" feeds pass 2 ──────
    done = threading.Event()
    t = threading.Thread(target=_spin, args=("htdemucs 4-stem", done), daemon=True)
    t.start()
    _, raw4 = sep4.separate_tensor(wav)
    done.set(); t.join()
    del sep4

    # ── Pass 2: htdemucs_6s on pass-1 "other" → Piano + Other ───────────
    print("[demucs]  Loading htdemucs_6s (6-stem: piano, bass, drums, vocals, guitar, other) ...")
    sep6 = Separator(model="htdemucs_6s", shifts=shifts, overlap=0.25)
    if sep6.samplerate != sr:
        other_in = torchaudio.functional.resample(raw4["other"], sr, sep6.samplerate)
    else:
        other_in = raw4["other"]
    print(f"[demucs]  Pass 2 (6-stem): extracting piano/other from 4-stem residual, shifts={shifts} ...")
    done = threading.Event()
    t = threading.Thread(target=_spin, args=("htdemucs_6s 6-stem", done), daemon=True)
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

    return stems, sr


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Pitch / note detection
# ─────────────────────────────────────────────────────────────────────────────

def _save_stem_wav(audio: np.ndarray, sr: int, path: Path):
    """Write a mono float32 array to a WAV file via soundfile."""
    try:
        import soundfile as sf
    except ImportError:
        sys.exit("soundfile not found.  Run: pip install soundfile")
    sf.write(str(path), audio, sr, subtype="FLOAT")


def detect_notes_piano(
    audio: np.ndarray,
    sr: int,
    tmp_dir: Path,
    min_confidence: float,
    min_duration_s: float,
) -> list[NoteEvent]:
    """
    Use piano_transcription_inference (Kong et al., MAESTRO-trained) to detect
    note events in a piano stem.  Returns fundamentals only — the model is
    explicitly trained to ignore partials.
    """
    try:
        from piano_transcription_inference import PianoTranscription
    except ImportError:
        print("[notes]   piano-transcription-inference not installed; "
              "skipping Piano overlay.  Run: pip install piano-transcription-inference")
        return []

    print("[notes]   Piano: running piano_transcription_inference ...")
    tmp_midi = str(tmp_dir / "_piano_tmp.mid")

    # Ensure model file is present.  The library uses wget (not on macOS by
    # default) and silently leaves the file missing if wget fails, so we handle
    # the download ourselves with urllib.  URL and local filename taken directly
    # from piano_transcription_inference/inference.py.
    _MODEL_DIR  = Path.home() / "piano_transcription_inference_data"
    _MODEL_FILE = _MODEL_DIR / "note_F1=0.9677_pedal_F1=0.9186.pth"
    if not _MODEL_FILE.exists() or _MODEL_FILE.stat().st_size < int(1.6e8):
        import urllib.request
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _url = ("https://zenodo.org/record/4034264/files/"
                "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1")
        print(f"[notes]   Downloading piano transcription model (~160 MB) -> {_MODEL_FILE}")
        urllib.request.urlretrieve(_url, str(_MODEL_FILE))

    # Resample to 16 kHz if needed — model expects 16k
    if sr != 16000:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    else:
        audio_16k = audio

    transcriptor = PianoTranscription(device="cpu", checkpoint_path=str(_MODEL_FILE))
    result = transcriptor.transcribe(audio_16k, tmp_midi)

    # est_note_events is a list of dicts:
    # {'onset_time': float, 'offset_time': float, 'midi_note': int, 'velocity': int}
    notes = []
    for event in result.get("est_note_events", []):
        onset    = float(event["onset_time"])
        offset   = float(event["offset_time"])
        midi_pitch = int(event["midi_note"])
        conf     = float(event["velocity"]) / 127.0
        if offset - onset < min_duration_s:
            continue
        if conf < min_confidence:
            continue
        notes.append(NoteEvent(onset, offset, midi_pitch, conf))

    print(f"[notes]   Piano: {len(notes)} notes detected "
          f"(confidence >= {min_confidence}, duration >= {min_duration_s}s)")
    return notes


def detect_notes_basic_pitch(
    stem_name: str,
    audio: np.ndarray,
    sr: int,
    tmp_dir: Path,
    min_confidence: float,
    min_duration_s: float,
) -> list[NoteEvent]:
    """
    Use Spotify basic-pitch to detect note events in a non-piano stem.
    basic-pitch accepts an audio file path, so we write a temp wav first.
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        print(f"[notes]   basic-pitch not installed; skipping {stem_name} overlay.  "
              "Run: pip install basic-pitch")
        return []

    tmp_wav = tmp_dir / f"_{stem_name.lower()}_tmp.wav"
    _save_stem_wav(audio, sr, tmp_wav)

    print(f"[notes]   {stem_name}: running basic-pitch ...")

    min_note_length_ms = int(min_duration_s * 1000)

    _, _, note_events = predict(
        str(tmp_wav),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=min_confidence,
        frame_threshold=0.3,
        minimum_note_length=min_note_length_ms,
        minimum_frequency=librosa.midi_to_hz(21),   # A0
        maximum_frequency=librosa.midi_to_hz(108),  # C8
    )

    # note_events: list of (start_s, end_s, pitch_midi, amplitude, pitch_bend)
    notes = [
        NoteEvent(float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3]))
        for ev in note_events
    ]

    print(f"[notes]   {stem_name}: {len(notes)} notes detected")
    return notes


def detect_notes(
    stem_name: str,
    audio: np.ndarray,
    sr: int,
    tmp_dir: Path,
    min_confidence: float,
    min_duration_s: float,
) -> list[NoteEvent]:
    """Dispatch to the right detector based on stem name."""
    if stem_name == "Piano":
        return detect_notes_piano(audio, sr, tmp_dir, min_confidence, min_duration_s)
    elif stem_name in ("Other", "Bass"):
        return detect_notes_basic_pitch(
            stem_name, audio, sr, tmp_dir, min_confidence, min_duration_s
        )
    else:
        return []   # Drums: no pitch detection


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: CQT computation
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
# Stage 5: Render
# ─────────────────────────────────────────────────────────────────────────────

BG         = "#111111"
GRID_C4    = "#ffffff"
DRUM_COLOR = "#aaaaaa"


def _draw_c4_reference(ax, duration: float, cfg: StaffConfig, pt_scale: float = 1.0):
    """
    Single C4 reference line + sparse faint octave lines.
    This is the only persistent pitch grid on pitched staves —
    actual note content is conveyed by the note-event overlay.
    """
    for octave in range(0, 9):
        hz = librosa.note_to_hz(f"C{octave}")
        if not (cfg.fmin_hz <= hz <= cfg.fmax_hz):
            continue
        is_c4 = (octave == 4)
        ax.axhline(
            y=hz,
            color="#ffffff" if is_c4 else "#333333",
            linewidth=(0.5 if is_c4 else 0.2) * pt_scale,
            alpha=0.9 if is_c4 else 0.6,
            zorder=2,
        )
        if is_c4:
            ax.text(
                duration * 0.001, hz, "C4",
                color="#ffffff", fontsize=4 * pt_scale, va="bottom",
                zorder=4, transform=ax.transData,
            )


def _draw_note_overlay(
    ax,
    notes: list[NoteEvent],
    cfg: StaffConfig,
    pt_scale: float = 1.0,
    note_lw: float = 1.5,
    px_per_sec: int = 60,
    staff_height_px: int = 280,
):
    """
    Draw each detected note as a horizontal bar at its fundamental frequency
    from onset to offset, with a pitch-name label at the onset.

    Collision avoidance — global 2D check across ALL labels:
      - Estimated text footprint in data units is computed from pt_scale,
        px_per_sec, and staff_height_px so thresholds are DPI-invariant.
      - A label is suppressed if any already-placed label is within
        time_gap seconds (x) AND pitch_gap semitones (y).
      - Bars are always drawn; only text is selectively skipped.
    """
    if not notes:
        return

    font_pts  = 3.0 * pt_scale            # rendered font size in points
    bar_lw    = note_lw * pt_scale

    # Estimate label footprint in data coordinates.
    # 1 pt = 1/72 inch; figure maps REF_DPI px → 1 inch, so 1 pt ≈ REF_DPI/72 px.
    # px_per_sec converts px → seconds for x; staff_height_px / n_octaves for y.
    n_octaves  = np.log2(cfg.fmax_hz / cfg.fmin_hz)
    ref_dpi    = 100
    pts_per_px = 72 / ref_dpi
    # ~4 chars wide, height = 1 line
    time_gap   = (font_pts / pts_per_px) * 4 / px_per_sec   # seconds
    pitch_gap  = (font_pts / pts_per_px) / (staff_height_px / (n_octaves * 12))  # semitones

    placed: list[tuple[float, int]] = []   # (onset_s, midi_pitch) of drawn labels

    for note in sorted(notes, key=lambda n: (n.onset_s, n.midi_pitch)):
        hz = librosa.midi_to_hz(note.midi_pitch)
        if not (cfg.fmin_hz <= hz <= cfg.fmax_hz):
            continue
        alpha = 0.4 + 0.6 * note.confidence
        ax.hlines(
            y=hz,
            xmin=note.onset_s,
            xmax=note.offset_s,
            colors=cfg.note_color,
            linewidths=bar_lw,
            alpha=alpha,
            zorder=5,
        )
        # Suppress label if it would overlap any previously placed label
        collides = any(
            abs(note.onset_s - px) < time_gap and abs(note.midi_pitch - pp) < pitch_gap
            for px, pp in placed
        )
        if not collides:
            ax.text(
                note.onset_s, hz,
                librosa.midi_to_note(note.midi_pitch),
                color=cfg.note_color, fontsize=font_pts,
                va="bottom", alpha=0.95,
                zorder=6, clip_on=True,
            )
            placed.append((note.onset_s, note.midi_pitch))


def _draw_drum_annotations(ax, duration: float, cfg: StaffConfig):
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
    note_events: dict[str, list[NoteEvent]],
    px_per_sec: int = 60,
    staff_height_px: int = 280,
    dpi: int = 100,
    note_overlay: bool = True,
):
    hop_length = max(1, int(sr / px_per_sec))
    duration = len(stems["Piano"]) / sr
    width_px  = int(duration * px_per_sec)
    total_height_px = staff_height_px * len(staff_configs)
    _REF_DPI  = 100
    pt_scale  = _REF_DPI / dpi   # keeps point-sized elements DPI-invariant

    print(f"[render]  Image: {width_px} x {total_height_px} px  "
          f"({duration:.1f}s @ {px_per_sec} px/s, dpi={dpi})")

    cqts = {}
    for cfg in staff_configs:
        print(f"[render]  CQT for {cfg.name} ...")
        done = threading.Event()
        t = threading.Thread(target=_spin, args=(f"CQT {cfg.name}", done), daemon=True)
        t.start()
        cqts[cfg.name] = compute_cqt_db(stems[cfg.name], sr, hop_length, cfg)
        done.set()
        t.join()

    fig = plt.figure(
        figsize=(width_px / dpi, total_height_px / dpi),
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

        if cfg.annotate_pitch:
            _draw_c4_reference(ax, duration, cfg, pt_scale=pt_scale)
            if note_overlay:
                _draw_note_overlay(ax, note_events.get(cfg.name, []), cfg,
                                   pt_scale=pt_scale, px_per_sec=px_per_sec,
                                   staff_height_px=staff_height_px)
            ax.set_yticklabels([])
        else:
            _draw_drum_annotations(ax, duration, cfg)
            ax.tick_params(axis="y", colors=DRUM_COLOR, labelsize=6)

        ax.set_ylabel(cfg.name, color="#cccccc", fontsize=8, rotation=90, labelpad=4)
        ax.yaxis.set_label_position("left")
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"[render]  Saved -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YouTube -> demucs -> 4-staff CQT score image with note overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url", nargs="?", default=None,
                        help="YouTube URL to download (omit if --file is given)")
    parser.add_argument("--file", "-f", default=None,
                        help="Path to an audio file already on disk (skips download)")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None,
                        help="Clip length in seconds from start")
    parser.add_argument("--out", "-o", default=None)
    parser.add_argument("--work-dir", default="./work")
    parser.add_argument("--px-per-sec", type=int, default=60)
    parser.add_argument("--staff-height", type=int, default=280)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--shifts", type=int, default=1,
                        help="Demucs test-time shifts (default 1; use 10 for best quality)")

    # Pitched staves
    parser.add_argument("--fmin", default="A0")
    parser.add_argument("--fmax", default="C8")
    parser.add_argument("--bins-per-octave", type=int, default=36)

    # Drums
    parser.add_argument("--drums-fmin", type=float, default=30.0)
    parser.add_argument("--drums-fmax", type=float, default=12000.0)
    parser.add_argument("--drums-bins-per-octave", type=int, default=36)

    # Note overlay
    parser.add_argument("--no-note-overlay", action="store_true",
                        help="Skip pitch detection; render pure CQT with C4 reference only")
    parser.add_argument("--note-confidence", type=float, default=0.5,
                        help="Minimum note confidence threshold 0.0–1.0 (default 0.5). "
                             "Raise to reduce spurious detections; lower to recover "
                             "quiet notes.")
    parser.add_argument("--min-note-duration", type=float, default=0.06,
                        help="Minimum note duration in seconds (default 0.06 = 60ms). "
                             "Filters sub-threshold transient false positives.")

    args = parser.parse_args()

    if args.file and args.url:
        parser.error("--file and url are mutually exclusive")
    if not args.file and not args.url:
        parser.error("provide a YouTube url or --file PATH")
    if args.duration is not None and (args.start is not None or args.end is not None):
        parser.error("--duration and --start/--end are mutually exclusive")
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be used together")

    pitched_fmin = librosa.note_to_hz(args.fmin)
    pitched_fmax = librosa.note_to_hz(args.fmax)

    staff_configs = [
        StaffConfig("Piano", "gray", annotate_pitch=True,
                    fmin_hz=pitched_fmin, fmax_hz=pitched_fmax,
                    bins_per_octave=args.bins_per_octave,
                    note_color="#00e5ff"),   # cyan
        StaffConfig("Other", "gray", annotate_pitch=True,
                    fmin_hz=pitched_fmin, fmax_hz=pitched_fmax,
                    bins_per_octave=args.bins_per_octave,
                    note_color="#ff9100"),   # amber
        StaffConfig("Bass",  "gray", annotate_pitch=True,
                    fmin_hz=pitched_fmin, fmax_hz=pitched_fmax,
                    bins_per_octave=args.bins_per_octave,
                    note_color="#eeff41"),   # lime
        StaffConfig("Drums", "gray", annotate_pitch=False,
                    fmin_hz=args.drums_fmin, fmax_hz=args.drums_fmax,
                    bins_per_octave=args.drums_bins_per_octave,
                    note_color="#ffffff"),   # unused for drums
    ]

    work_dir = Path(args.work_dir)

    # Stage 1: Source
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

    stems, sr = separate_stems(wav_path, start_s=seg_start, end_s=seg_end, shifts=args.shifts)

    # Stage 3: Pitch detection
    note_events: dict[str, list[NoteEvent]] = {}
    if not args.no_note_overlay:
        tmp_dir = work_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for cfg in staff_configs:
            if cfg.annotate_pitch:
                note_events[cfg.name] = detect_notes(
                    cfg.name,
                    stems[cfg.name],
                    sr,
                    tmp_dir,
                    min_confidence=args.note_confidence,
                    min_duration_s=args.min_note_duration,
                )

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
        note_events=note_events,
        px_per_sec=args.px_per_sec,
        staff_height_px=args.staff_height,
        dpi=args.dpi,
        note_overlay=not args.no_note_overlay,
    )


if __name__ == "__main__":
    main()