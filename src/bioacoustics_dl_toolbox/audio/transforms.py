"""
Audio and spectral transforms.

All transforms are callable objects that can be composed via ``Compose``.
Ported from ANIMAL-SPOT ``data/transforms.py`` (Bergler & Schroeter, GPL-3.0).
"""

from __future__ import annotations

import io
import math
import os
from multiprocessing import Lock
from typing import Any, Callable, List

import numpy as np
import scipy.fftpack
import torch
import torch.nn.functional as F

from bioacoustics_dl_toolbox.audio.io import load_audio_file
from bioacoustics_dl_toolbox.io.async_file import AsyncFileReader, AsyncFileWriter


class Compose:
    """Composes several transforms into one."""

    def __init__(self, *transforms: Any) -> None:
        if len(transforms) == 1 and isinstance(transforms[0], list):
            self.transforms: list[Any] = transforms[0]
        else:
            self.transforms = list(transforms)

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class SqueezeDim0:
    """Squeeze the tensor at dim=0."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=0)


class UnsqueezeDim0:
    """Unsqueeze the tensor at dim=0."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(dim=0)


class ToFloatTensor:
    """Convert a numpy array to a ``torch.FloatTensor``."""

    def __call__(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        elif isinstance(x, torch.Tensor):
            return x.float()
        raise ValueError("Unknown input array type: {}".format(type(x)))


class ToFloatNumpy:
    """Convert a tensor to a float32 numpy array."""

    def __call__(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x.astype("float32")
        elif isinstance(x, torch.Tensor):
            return x.float().numpy()
        raise ValueError("Unknown input array type: {}".format(type(x)))


class PreEmphasize:
    """Pre-emphasis filter to boost higher frequencies.

    Parameters
    ----------
    factor:
        Pre-emphasis coefficient (typically 0.97).
    """

    def __init__(self, factor: float = 0.97) -> None:
        self.factor = factor

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2:
            raise ValueError(
                "PreEmphasize expects a 2-dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        return torch.cat(
            (y[:, 0].unsqueeze(dim=-1), y[:, 1:] - self.factor * y[:, :-1]), dim=-1
        )


class Spectrogram:
    """Compute a power spectrogram from a waveform tensor.

    Parameters
    ----------
    n_fft:
        FFT window size.
    hop_length:
        Hop length between frames.
    center:
        Whether to pad the signal so that frame ``t`` is centered at
        ``t * hop_length``.
    """

    def __init__(self, n_fft: int, hop_length: int, center: bool = True) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.n_fft)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2-dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )

        y = _ensure_min_signal_length(y, min_length=self.n_fft)

        window = self.window.to(device=y.device, dtype=y.dtype)

        stft_result = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=self.center,
            onesided=True,
            return_complex=True,
        ).transpose(1, 2)

        stft_result /= window.pow(2).sum().sqrt()
        stft_result = stft_result.abs().pow(2)
        return stft_result


class CachedSpectrogram:
    """Compute spectrograms with on-disk caching.

    Parameters
    ----------
    cache_dir:
        Directory to store cached spectrogram files.
    spec_transform:
        The spectrogram transform to apply when cache misses.
    file_reader:
        Optional ``AsyncFileReader`` instance.
    file_writer:
        Optional ``AsyncFileWriter`` instance.
    **meta:
        Additional metadata keys to validate cache entries against.
    """

    version: int = 4

    def __init__(
        self,
        cache_dir: str,
        spec_transform: Callable[..., torch.Tensor],
        file_reader: AsyncFileReader | None = None,
        file_writer: AsyncFileWriter | None = None,
        **meta: Any,
    ) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.reader = file_reader if file_reader is not None else AsyncFileReader(n_readers=1)
        self.transform = spec_transform
        self.meta = meta
        self.writer = (
            file_writer
            if file_writer is not None
            else AsyncFileWriter(write_fn=self._write_fn, n_writers=1)
        )

    def get_cached_name(self, file_name: str) -> str:
        cached_spec_name = os.path.splitext(os.path.basename(file_name))[0] + ".spec"
        dir_structure = os.path.dirname(file_name).replace(r"/", "_") + "_"
        cached_spec_name = dir_structure + cached_spec_name
        if not os.path.isabs(cached_spec_name):
            cached_spec_name = os.path.join(self.cache_dir, cached_spec_name)
        return cached_spec_name

    def __call__(self, file_name: str) -> torch.Tensor:
        cached_spec_name = self.get_cached_name(file_name)
        if not os.path.isfile(cached_spec_name):
            return self._compute_and_cache(file_name)
        try:
            data = self.reader(cached_spec_name)
            spec_dict = torch.load(io.BytesIO(data), map_location="cpu")  # type: ignore[arg-type]
        except (EOFError, RuntimeError):
            return self._compute_and_cache(file_name)
        if not (
            "v" in spec_dict
            and spec_dict["v"] == self.version
            and "data" in spec_dict
            and spec_dict["data"].dim() == 3
        ):
            return self._compute_and_cache(file_name)
        for key, value in self.meta.items():
            if not (key in spec_dict and spec_dict[key] == value):
                return self._compute_and_cache(file_name)
        return spec_dict["data"]  # type: ignore[no-any-return]

    def _compute_and_cache(self, file_name: str) -> torch.Tensor:
        try:
            audio_data = self.reader(file_name)
            spec = self.transform(io.BytesIO(audio_data))
        except Exception:
            spec = self.transform(file_name)
        self.writer(self.get_cached_name(file_name), spec)
        return spec  # type: ignore[no-any-return]

    def _write_fn(self, file_name: str, data: torch.Tensor) -> None:
        spec_dict: dict[str, Any] = {"v": self.version, "data": data}
        for key, value in self.meta.items():
            spec_dict[key] = value
        torch.save(spec_dict, file_name)


class MeanStdNormalize:
    """Normalize a spectrogram to zero mean and unit variance."""

    def __call__(
        self, spectrogram: torch.Tensor, ret_dict: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        mean = spectrogram.mean()
        spectrogram.sub_(mean)
        std = spectrogram.std()
        spectrogram.div_(std)
        if ret_dict is not None:
            ret_dict["mean"] = mean
            ret_dict["std"] = std
        return spectrogram


class Normalize:
    """dB-normalize a spectrogram to the range [0, 1]."""

    def __init__(self, min_level_db: float = -100, ref_level_db: float = 20) -> None:
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (spec - self.ref_level_db - self.min_level_db) / -self.min_level_db, 0, 1
        )


class MinMaxNormalize:
    """Min-max normalize a spectrogram to the range [0, 1]."""

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram -= spectrogram.min()
        if spectrogram.max().item() == 0.0:
            return spectrogram
        spectrogram /= spectrogram.max()
        return spectrogram


class Amp2Db:
    """Convert a power or amplitude spectrogram to decibel scale.

    Based on code from ``torchaudio`` (BSD-2-Clause, Facebook Inc.).
    Modified by Bergler & Schroeter.
    """

    def __init__(
        self, min_level_db: float | None = None, stype: str = "power"
    ) -> None:
        self.stype = stype
        self.multiplier = 10.0 if stype == "power" else 20.0
        if min_level_db is None:
            self.min_level: torch.Tensor | None = None
        else:
            min_level_db = -min_level_db if min_level_db > 0 else min_level_db
            self.min_level = torch.tensor(
                np.exp(min_level_db / self.multiplier * np.log(10))
            )

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        if self.min_level is not None:
            spec_ = torch.max(spec, self.min_level)
        else:
            spec_ = spec
        return self.multiplier * torch.log10(spec_)


class SPECLOG1P:
    """Compress a spectrogram using ``log1p(spec * factor)``."""

    def __init__(self, compression_factor: int = 1) -> None:
        self.compression_factor = compression_factor

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return torch.log1p(spectrogram * self.compression_factor)


class SPECEXPM1:
    """Decompress a spectrogram using ``expm1(spec) / factor``."""

    def __init__(self, decompression_factor: int = 1) -> None:
        self.decompression_factor = decompression_factor

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return torch.expm1(spectrogram) / self.decompression_factor


def _scale(spectrogram: torch.Tensor, shift_factor: float, dim: int) -> torch.Tensor:
    """Scale a spectrogram dimension (time or frequency) by a given factor."""
    in_dim = spectrogram.dim()
    if in_dim < 3:
        raise ValueError(
            "Expected spectrogram with size (c t f) or (n c t f)"
            ", but got {}".format(spectrogram.size())
        )
    if in_dim == 3:
        spectrogram.unsqueeze_(dim=0)
    size = list(spectrogram.shape)[2:]
    dim -= 1
    size[dim] = int(round(size[dim] * shift_factor))
    spectrogram = F.interpolate(spectrogram, size=size, mode="nearest")
    if in_dim == 3:
        spectrogram.squeeze_(dim=0)
    return spectrogram


class RandomPitchShift:
    """Randomly shift pitch by scaling along the frequency axis.

    The shift factor is ``2**Uniform(log2(from_), log2(to_))``.
    """

    def __init__(self, from_: float = 0.5, to_: float = 1.5) -> None:
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        median = spectrogram.median()
        size = list(spectrogram.shape)
        scaled = _scale(spectrogram, factor, dim=2)
        if factor > 1:
            out = scaled[:, :, : size[2]]
        else:
            out = torch.full(size, fill_value=median, dtype=spectrogram.dtype)
            new_f_bins = int(round(size[2] * factor))
            out[:, :, 0:new_f_bins] = scaled
        return out


class RandomTimeStretch:
    """Randomly stretch time by scaling along the time axis.

    The stretch factor is ``2**Uniform(log2(from_), log2(to_))``.
    """

    def __init__(self, from_: float = 0.5, to_: float = 2.0) -> None:
        self.from_ = math.log2(from_)
        self.to_ = math.log2(to_)

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        factor = 2 ** torch.empty((1,)).uniform_(self.from_, self.to_).item()
        return _scale(spectrogram, factor, dim=1)


class RandomAmplitude:
    """Randomly scale amplitude in dB."""

    def __init__(self, increase_db: int = 3, decrease_db: int | None = None) -> None:
        self.increase_db = increase_db
        if decrease_db is None:
            decrease_db = -increase_db
        elif decrease_db > 0:
            decrease_db *= -1
        self.decrease_db = decrease_db

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        db_change = torch.randint(
            self.decrease_db, self.increase_db, size=(1,), dtype=torch.float
        )
        return spec.mul(10 ** (db_change / 10))


class RandomAddNoise:
    """Add random noise from a bank of noise files at a random SNR.

    The noise spectrogram is computed on the fly, optionally augmented,
    then mixed with the input at a random SNR between ``max_snr`` and
    ``min_snr`` dB.
    """

    def __init__(
        self,
        noise_files: List[str],
        spectrogram_transform: Callable[..., torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
        min_length: int = 0,
        min_snr: int = 12,
        max_snr: int = -3,
        return_original: bool = False,
    ) -> None:
        if not noise_files:
            raise ValueError("No noise files found")
        self.noise_files = noise_files
        self.spectrogram_transform = spectrogram_transform
        self.noise_file_locks = {file: Lock() for file in noise_files}
        self.transform = transform
        self.min_length = min_length
        self.pad_sampler = PaddedSubsequenceSampler(sequence_length=min_length, dim=1)
        self.min_snr = min_snr if min_snr > max_snr else max_snr
        self.max_snr = max_snr if min_snr > max_snr else min_snr
        self.return_original = return_original

    def __call__(
        self, spectrogram: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if len(self.noise_files) == 1:
            idx = 0
        else:
            idx = torch.randint(
                0, len(self.noise_files), size=(1,), dtype=torch.long
            ).item()
        noise_file = self.noise_files[idx]
        lock = self.noise_file_locks[noise_file]
        acquired = lock.acquire(timeout=10)
        if not acquired:
            print("Warning: Could not acquire lock for {}".format(noise_file))
            return spectrogram
        try:
            noise_spec = self.spectrogram_transform(noise_file)
        except Exception:
            import traceback
            print(traceback.format_exc())
            return spectrogram
        finally:
            lock.release()

        noise_spec = self.pad_sampler._maybe_sample_subsequence(
            noise_spec, spectrogram.size(1) * 2
        )
        noise_spec = self.transform(noise_spec)

        if self.min_length > 0:
            spectrogram = self.pad_sampler._maybe_pad(spectrogram)

        if spectrogram.size(1) > noise_spec.size(1):
            n_repeat = int(math.ceil(spectrogram.size(1) / noise_spec.size(1)))
            noise_spec = noise_spec.repeat(1, n_repeat, 1)

        if spectrogram.size(1) < noise_spec.size(1):
            high = noise_spec.size(1) - spectrogram.size(1)
            start = torch.randint(0, high, size=(1,), dtype=torch.long)
            end = start + spectrogram.size(1)
            noise_spec_part = noise_spec[:, start:end]
        else:
            noise_spec_part = noise_spec

        snr = torch.randint(self.max_snr, self.min_snr, size=(1,), dtype=torch.float)
        signal_power = spectrogram.sum()
        noise_power = noise_spec_part.sum()
        gain = (signal_power / noise_power) * 10 ** (-snr / 10)
        spectrogram_augmented = spectrogram + noise_spec_part * gain

        if self.return_original:
            return spectrogram_augmented, spectrogram
        return spectrogram_augmented


class PaddedSubsequenceSampler:
    """Sample a fixed-length subsequence along one axis, padding if necessary."""

    def __init__(
        self, sequence_length: int, dim: int = 0, random: bool = True
    ) -> None:
        assert isinstance(sequence_length, int)
        assert isinstance(dim, int)
        self.sequence_length = sequence_length
        self.dim = dim
        if random:
            self._sampler: Callable[[int], int] = lambda x: torch.randint(
                0, x, size=(1,), dtype=torch.long
            ).item()
        else:
            self._sampler = lambda x: x // 2

    def _maybe_sample_subsequence(
        self, spectrogram: torch.Tensor, sequence_length: int | None = None
    ) -> torch.Tensor:
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length > sequence_length:
            start = self._sampler(sample_length - sequence_length)
            end = start + sequence_length
            indices = torch.arange(start, end, dtype=torch.long)
            return torch.index_select(spectrogram, self.dim, indices)
        return spectrogram

    def _maybe_pad(
        self, spectrogram: torch.Tensor, sequence_length: int | None = None
    ) -> torch.Tensor:
        if sequence_length is None:
            sequence_length = self.sequence_length
        sample_length = spectrogram.shape[self.dim]
        if sample_length < sequence_length:
            start = self._sampler(sequence_length - sample_length)
            end = start + sample_length
            shape = list(spectrogram.shape)
            shape[self.dim] = sequence_length
            padded_spectrogram = torch.zeros(shape, dtype=spectrogram.dtype)
            if self.dim == 0:
                padded_spectrogram[start:end] = spectrogram
            elif self.dim == 1:
                padded_spectrogram[:, start:end] = spectrogram
            elif self.dim == 2:
                padded_spectrogram[:, :, start:end] = spectrogram
            elif self.dim == 3:
                padded_spectrogram[:, :, :, start:end] = spectrogram
            return padded_spectrogram
        return spectrogram

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = self._maybe_pad(spectrogram)
        spectrogram = self._maybe_sample_subsequence(spectrogram)
        return spectrogram


class Interpolate:
    """Frequency compression via interpolation into a target number of bins."""

    def __init__(
        self,
        n_freqs: int,
        sample_rate: int | None = None,
        f_min: int = 0,
        f_max: int | None = None,
    ) -> None:
        self.n_freqs = n_freqs
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        n_fft = (spec.size(2) - 1) * 2
        if self.sample_rate is not None and n_fft is not None:
            min_bin = int(max(0, math.floor(n_fft * self.f_min / self.sample_rate)))
            max_bin = int(min(n_fft - 1, math.ceil(n_fft * self.f_max / self.sample_rate)))  # type: ignore[arg-type]
            spec = spec[:, :, min_bin:max_bin]
        spec.unsqueeze_(dim=0)
        spec = F.interpolate(spec, size=(spec.size(2), self.n_freqs), mode="nearest")
        return spec.squeeze(dim=0)


def _hz2mel(f: float) -> float:
    return 2595 * np.log10(1 + f / 700)


def _mel2hz(mel: torch.Tensor | float) -> Any:
    return 700 * (10 ** (mel / 2595) - 1)


def _melbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 128,
    f_min: float = 0.0,
    f_max: float | None = None,
    inverse: bool = False,
) -> torch.Tensor:
    """Create a mel filterbank matrix.

    Based on code from ``torchaudio`` (BSD-2-Clause, Facebook Inc.).
    Modified by Bergler & Schroeter.
    """
    m_min = 0.0 if f_min == 0 else _hz2mel(f_min)
    m_max = _hz2mel(f_max if f_max is not None else sample_rate // 2)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel2hz(m_pts)
    bins = torch.floor(((n_fft - 1) * 2 + 1) * f_pts / sample_rate).long()
    fb = torch.zeros(n_mels, n_fft)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1].item()
        f_m = bins[m].item()
        f_m_plus = bins[m + 1].item()
        if f_m_minus != f_m:
            fb[m - 1, f_m_minus:f_m] = (
                torch.arange(f_m_minus, f_m) - f_m_minus
            ).float() / (f_m - f_m_minus)
        if f_m != f_m_plus:
            fb[m - 1, f_m:f_m_plus] = (
                f_m_plus - torch.arange(f_m, f_m_plus)
            ).float() / (f_m_plus - f_m)
    if not inverse:
        return fb.t()
    return fb


class F2M:
    """Convert a linear-frequency spectrogram to mel scale.

    Based on code from ``torchaudio`` (BSD-2-Clause, Facebook Inc.).
    Modified by Bergler & Schroeter.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_mels: int = 40,
        f_min: int = 0,
        f_max: int | None = None,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_f: torch.Tensor) -> torch.Tensor:
        n_fft = spec_f.size(2)
        fb = _melbank(self.sr, n_fft, self.n_mels, self.f_min, self.f_max)
        return torch.matmul(spec_f, fb)


class M2F:
    """Convert a mel spectrogram back to linear frequency (approximate inverse)."""

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 1024,
        f_min: int = 0,
        f_max: int | None = None,
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft // 2 + 1
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sr // 2
        if self.f_max > self.sr // 2:
            raise ValueError("f_max > sr // 2")

    def __call__(self, spec_m: torch.Tensor) -> torch.Tensor:
        n_mels = spec_m.size(2)
        fb = _melbank(self.sr, self.n_fft, n_mels, self.f_min, self.f_max, inverse=True)
        return torch.matmul(spec_m, fb)


class M2MFCC:
    """Convert a mel spectrogram to MFCCs via DCT."""

    def __init__(self, n_mfcc: int = 32) -> None:
        self.n_mfcc = n_mfcc

    def __call__(self, spec_m: torch.Tensor) -> torch.Tensor:
        device = spec_m.device
        spec_m = 10 * torch.log10(spec_m)
        spec_m[spec_m == float("-inf")] = 0
        if isinstance(spec_m, torch.Tensor):
            spec_m_np = spec_m.cpu().numpy()
        else:
            spec_m_np = spec_m
        mfcc = scipy.fftpack.dct(spec_m_np, axis=-1)
        mfcc = mfcc[:, :, 1 : self.n_mfcc + 1]
        return torch.from_numpy(mfcc).to(device)


def _ensure_min_signal_length(signal: torch.Tensor, min_length: int) -> torch.Tensor:
    if signal.dim() != 2:
        raise ValueError(
            "Expected signal with size (c, n), but got size: {}.".format(signal.size())
        )

    if min_length <= 0:
        raise ValueError("min_length must be > 0, but got {}.".format(min_length))

    current_length = int(signal.size(1))
    if current_length >= min_length:
        return signal

    if current_length == 0:
        return torch.zeros(
            (int(signal.size(0)), min_length),
            dtype=signal.dtype,
            device=signal.device,
        )

    repeats = int(math.ceil(min_length / current_length))
    tiled = signal.repeat(1, repeats)
    extra = int(tiled.size(1) - min_length)
    start_index = extra // 2
    end_index = start_index + min_length
    return tiled[:, start_index:end_index]
