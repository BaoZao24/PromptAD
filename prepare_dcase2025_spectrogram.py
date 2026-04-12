"""Convert DCASE2025 wav files to FBank segment images.

The script keeps the input directory structure and writes PNG files to a
mirrored output tree. It is designed for the current DCASE2025 layout under
PromptAD/datasets/DCASE2025.
"""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 1024
DEFAULT_WIN_MS = 25.0
DEFAULT_HOP_MS = 10.0
DEFAULT_N_MELS = 128
DEFAULT_SEGMENT_SIZE = 128
DEFAULT_TRAIN_HOP = 128
DEFAULT_TEST_HOP = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert wav files to FBank PNGs.")
    parser.add_argument(
        "input_root",
        nargs="?",
        default="/home/wangbei/experiment/anomaly_detection/PromptAD/datasets/DCASE2025",
        help="Root directory that contains the wav files.",
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        default=None,
        help="Output directory for FBank PNGs. Defaults to <input_root>_fbank.",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Target sample rate.")
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT, help="FFT size for STFT.")
    parser.add_argument("--win-ms", type=float, default=DEFAULT_WIN_MS, help="Window length in milliseconds.")
    parser.add_argument("--hop-ms", type=float, default=DEFAULT_HOP_MS, help="Frame shift in milliseconds.")
    parser.add_argument("--n-mels", type=int, default=DEFAULT_N_MELS, help="Number of mel bins.")
    parser.add_argument("--segment-size", type=int, default=DEFAULT_SEGMENT_SIZE, help="Time-frequency segment size.")
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Segmentation hop policy: train uses 128, test uses 5.",
    )
    parser.add_argument("--segment-hop", type=int, default=None, help="Override segmentation hop size.")
    parser.add_argument("--window", type=str, default="hann", help="Window type for STFT.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="Output image extension. Only PNG is used by default.",
    )
    return parser.parse_args()


def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(str(path))
    source_dtype = audio.dtype

    if np.issubdtype(source_dtype, np.integer):
        max_abs = float(np.iinfo(source_dtype).max)
        if max_abs > 0:
            audio = audio / max_abs

    audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return sample_rate, audio


def maybe_resample(audio: np.ndarray, source_rate: int, target_rate: int | None) -> np.ndarray:
    if target_rate is None or target_rate == source_rate:
        return audio

    if len(audio) == 0:
        return audio

    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(freq_mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    mel_min = hz_to_mel(np.array([0.0], dtype=np.float32))[0]
    mel_max = hz_to_mel(np.array([sample_rate / 2.0], dtype=np.float32))[0]
    mel_points = np.linspace(mel_min, mel_max, num=n_mels + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for mel_idx in range(1, n_mels + 1):
        left = bin_points[mel_idx - 1]
        center = bin_points[mel_idx]
        right = bin_points[mel_idx + 1]

        left = max(left, 0)
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_freqs)

        for freq_idx in range(left, min(center, n_freqs)):
            filterbank[mel_idx - 1, freq_idx] = (freq_idx - left) / max(center - left, 1)
        for freq_idx in range(center, right):
            filterbank[mel_idx - 1, freq_idx] = (right - freq_idx) / max(right - center, 1)

    return filterbank


def compute_fbank(
    audio: np.ndarray,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
    sample_rate: int,
    n_mels: int,
) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    win_length = min(win_length, n_fft)
    window_values = get_window(window, win_length, fftbins=True).astype(np.float32)

    if len(audio) < win_length:
        pad_width = win_length - len(audio)
        audio = np.pad(audio, (0, pad_width), mode="constant")

    frame_count = 1 + int(np.ceil(max(0, len(audio) - win_length) / hop_length))
    total_length = (frame_count - 1) * hop_length + win_length
    if len(audio) < total_length:
        audio = np.pad(audio, (0, total_length - len(audio)), mode="constant")

    frame_indices = np.arange(win_length)[None, :] + hop_length * np.arange(frame_count)[:, None]
    frames = audio[frame_indices] * window_values[None, :]
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power_spectrum = (np.abs(spectrum) ** 2).T

    mel_filterbank = build_mel_filterbank(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels)
    fbank = np.dot(mel_filterbank, power_spectrum)
    log_fbank = np.log(fbank + 1e-10)

    fbank_min = log_fbank.min()
    fbank_max = log_fbank.max()
    if np.isclose(fbank_min, fbank_max):
        normalized = np.zeros_like(log_fbank, dtype=np.float32)
    else:
        normalized = (log_fbank - fbank_min) / (fbank_max - fbank_min)

    return normalized.astype(np.float32)


def segment_fbank(fbank: np.ndarray, segment_size: int, segment_hop: int) -> np.ndarray:
    freq_bins, time_steps = fbank.shape
    if time_steps < segment_size:
        pad_width = segment_size - time_steps
        fbank = np.pad(fbank, ((0, 0), (0, pad_width)), mode="constant")
        time_steps = fbank.shape[1]

    if time_steps == segment_size:
        starts = [0]
    else:
        starts = list(range(0, max(time_steps - segment_size + 1, 1), segment_hop))
        last_start = time_steps - segment_size
        if starts[-1] != last_start:
            starts.append(last_start)

    segments = [fbank[:, start : start + segment_size] for start in starts]
    return np.stack(segments, axis=0)


def fbank_segments_to_images(segments: np.ndarray) -> Iterable[np.ndarray]:
    for segment in segments:
        yield (np.clip(segment, 0.0, 1.0) * 255.0).astype(np.uint8)


def infer_split_from_path(path: Path, default_phase: str) -> str:
    if 'train' in path.parts:
        return 'train'
    if 'test' in path.parts:
        return 'test'
    return default_phase


def convert_file(
    wav_path: Path,
    input_root: Path,
    output_root: Path,
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
    n_mels: int,
    segment_size: int,
    segment_hop: int,
    overwrite: bool,
    ext: str,
) -> Path:
    relative_path = wav_path.relative_to(input_root)
    source_rate, audio = read_wav(wav_path)
    audio = maybe_resample(audio, source_rate, sample_rate)
    fbank = compute_fbank(
        audio=audio,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        sample_rate=sample_rate,
        n_mels=n_mels,
    )
    segments = segment_fbank(fbank, segment_size=segment_size, segment_hop=segment_hop)

    last_output_path = output_root / relative_path.parent
    last_output_path.mkdir(parents=True, exist_ok=True)
    stem = relative_path.stem

    for segment_index, image in enumerate(fbank_segments_to_images(segments)):
        output_path = last_output_path / f"{stem}_seg_{segment_index:04d}.{ext.lower()}"
        if output_path.exists() and not overwrite:
            continue
        Image.fromarray(image).save(output_path)

    return last_output_path


def iter_wavs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.wav")):
        if path.is_file():
            yield path
    for path in sorted(root.rglob("*.WAV")):
        if path.is_file():
            yield path


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else input_root.parent / f"{input_root.name}_fbank"
    )

    win_length = max(1, int(round(args.sample_rate * args.win_ms / 1000.0)))
    hop_length = max(1, int(round(args.sample_rate * args.hop_ms / 1000.0)))
    segment_hop = args.segment_hop if args.segment_hop is not None else (DEFAULT_TRAIN_HOP if args.phase == "train" else DEFAULT_TEST_HOP)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    wav_files = list(iter_wavs(input_root))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found under: {input_root}")

    converted = 0
    skipped = 0
    for wav_path in wav_files:
        split = infer_split_from_path(wav_path, args.phase)
        current_segment_hop = args.segment_hop if args.segment_hop is not None else (DEFAULT_TRAIN_HOP if split == 'train' else DEFAULT_TEST_HOP)

        converted_folder = output_root / wav_path.relative_to(input_root).parent
        expected_prefix = wav_path.stem + "_seg_"
        if converted_folder.exists() and not args.overwrite:
            existing_files = list(converted_folder.glob(f"{expected_prefix}*.{args.ext.lower()}"))
            if existing_files:
                skipped += 1
                continue

        convert_file(
            wav_path=wav_path,
            input_root=input_root,
            output_root=output_root,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=args.window,
            n_mels=args.n_mels,
            segment_size=args.segment_size,
            segment_hop=current_segment_hop,
            overwrite=args.overwrite,
            ext=args.ext,
        )
        converted += 1

    print(f"Input root:  {input_root}")
    print(f"Output root: {output_root}")
    print(f"Converted:   {converted}")
    print(f"Skipped:      {skipped}")
    print(
        f"FBank: sample_rate={args.sample_rate}, n_fft={args.n_fft}, win_length={win_length}, "
        f"hop_length={hop_length}, n_mels={args.n_mels}, window={args.window}"
    )
    print(f"Segment: segment_size={args.segment_size}, train_hop={DEFAULT_TRAIN_HOP}, test_hop={DEFAULT_TEST_HOP}, phase={args.phase}")


if __name__ == "__main__":
    main()