#!/usr/bin/env python
"""Generate paired target-scene OFDMA cold-start observations.

The upstream simulator samples a new legitimate configuration for every
observation.  This entry point instead samples one legitimate configuration
per target scene, traces its legitimate channel once, and renders multiple
normal/support and normal/anomalous test observations from that fixed context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ofdma_target_scene_coldstart_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def array_hash(arrays: Iterable[np.ndarray], length: int = 16) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()[:length]


def derived_seed(base_seed: int, split: str, scene_index: int, stream: int, item: int = 0) -> int:
    split_code = {"validation": 1, "test": 2, "smoke": 3}[split]
    sequence = np.random.SeedSequence(
        [int(base_seed), split_code, int(scene_index), int(stream), int(item)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def parse_scene_indexes(value: str | None, count: int) -> list[int]:
    if value is None:
        return list(range(count))
    indexes: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid scene range: {part}")
            indexes.update(range(start, end + 1))
        else:
            indexes.add(int(part))
    result = sorted(indexes)
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError(f"Scene indexes must fall in [0, {count - 1}], got {result}")
    return result


@dataclass
class SimulatorRuntime:
    config: Any
    scene: Any
    scene_size: Any
    path_solver: Any
    fft_frequencies: Any
    jammer_pattern: str
    su_coordinates: Any
    rt_utils: Any
    ofdm_utils: Any
    Transmitter: Any
    Jammer: Any
    Sample: Any
    srt: Any


def initialize_simulator(protocol: dict[str, Any]) -> SimulatorRuntime:
    simulator_root = Path(protocol["simulator_root"]).resolve()
    source_root = simulator_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing OFDMA simulator source: {source_root}")
    sys.path.insert(0, str(source_root))

    import sionna.rt as srt
    from omegaconf import OmegaConf, open_dict
    from utils import ofdm_utils, rt_utils
    from utils.datatypes import Jammer, Sample, Transmitter

    config = OmegaConf.load(source_root / "conf" / "dataset_generation.yaml")
    context_cfg = protocol["normal_context"]
    config.min_tx = int(context_cfg["min_legitimate_transmitters"])
    config.max_tx = int(context_cfg["max_legitimate_transmitters"])

    su_coordinates = rt_utils.get_su_coordinates(str(simulator_root))
    scene_path = simulator_root / "scenes" / f"scene{config.scene_nr}" / "scene.xml"
    scene = rt_utils.init_scene(str(scene_path), config.f_c, su_coordinates)
    scene_size = rt_utils.get_scene_size(scene)

    if config.jam_pattern == "iso":
        jammer_pattern = "iso"
    elif config.jam_pattern == "directional":
        jammer_pattern = "tr38901"
    else:
        raise ValueError(f"Unsupported jammer pattern: {config.jam_pattern}")

    if config.bandwidth == 20e6 and config.subcarrier_spacing == 15e3:
        num_rb = 110
    else:
        num_rb = (
            int(config.bandwidth / config.subcarrier_spacing)
            // config.subcarriers_per_rb
        )
    num_subcarriers = int(num_rb * config.subcarriers_per_rb)
    idx_first_sc = (config.nfft - num_subcarriers) // 2
    fft_frequencies = (
        srt.utils.subcarrier_frequencies(config.nfft, config.subcarrier_spacing)
        + config.f_c
    )
    p_noise_dbm = float(
        rt_utils.calc_noise_power_dbm(
            config.nfft * config.subcarrier_spacing,
            config.su_noise_figure,
            config.additional_impairments,
        )
    )
    with open_dict(config):
        config.num_rb = num_rb
        config.num_subcarriers = num_subcarriers
        config.idx_first_sc = idx_first_sc
        config.p_noise_dbm = p_noise_dbm

    return SimulatorRuntime(
        config=config,
        scene=scene,
        scene_size=scene_size,
        path_solver=srt.PathSolver(),
        fft_frequencies=fft_frequencies,
        jammer_pattern=jammer_pattern,
        su_coordinates=su_coordinates,
        rt_utils=rt_utils,
        ofdm_utils=ofdm_utils,
        Transmitter=Transmitter,
        Jammer=Jammer,
        Sample=Sample,
        srt=srt,
    )


def sample_frequency_bands(
    rng: np.random.Generator, num_transmitters: int, num_resource_blocks: int
) -> list[dict[str, int]]:
    """Create non-overlapping fixed frequency bands for one target scene."""
    maximum_width = max(1, int((num_resource_blocks // num_transmitters) * 1.3))
    widths = rng.integers(1, maximum_width + 1, size=num_transmitters)
    if int(widths.sum()) > num_resource_blocks:
        scaled = np.maximum(
            1,
            np.floor(widths * (num_resource_blocks / float(widths.sum()))).astype(int),
        )
        widths = scaled
        while int(widths.sum()) > num_resource_blocks:
            widest = int(np.argmax(widths))
            if widths[widest] <= 1:
                raise RuntimeError("Could not fit transmitter bands into the OFDMA grid")
            widths[widest] -= 1

    remaining = int(num_resource_blocks - widths.sum())
    gap_weights = rng.dirichlet(np.ones(num_transmitters + 1))
    gaps = rng.multinomial(remaining, gap_weights)
    bands: list[dict[str, int]] = []
    cursor = int(gaps[0])
    for tx_index, width in enumerate(widths.tolist()):
        bands.append({"start_rb": cursor, "end_rb": cursor + int(width)})
        cursor += int(width) + int(gaps[tx_index + 1])
    return bands


def sample_target_context(
    runtime: SimulatorRuntime,
    protocol: dict[str, Any],
    split: str,
    scene_index: int,
) -> dict[str, Any]:
    context_seed = derived_seed(protocol["base_seed"], split, scene_index, stream=1)
    rng = np.random.default_rng(context_seed)
    num_transmitters = int(
        rng.integers(runtime.config.min_tx, runtime.config.max_tx + 1)
    )
    bands = sample_frequency_bands(rng, num_transmitters, runtime.config.num_rb)

    np.random.seed(context_seed)
    positions = [
        [
            float(value)
            for value in runtime.rt_utils.get_random_tx_location(
                runtime.config.scene_nr,
                runtime.scene_size,
                tx_height=runtime.config.sionna.tx_height,
            )
        ]
        for _ in range(num_transmitters)
    ]
    context = {
        "target_scene_id": (
            "smoke_000"
            if split == "smoke"
            else f"{'val' if split == 'validation' else 'test'}_{scene_index:03d}"
        ),
        "split": split,
        "scene_index": scene_index,
        "context_seed": context_seed,
        "physical_scene_number": int(runtime.config.scene_nr),
        "num_legitimate_transmitters": num_transmitters,
        "legitimate_transmitter_positions": positions,
        "legitimate_frequency_bands": bands,
    }
    context["context_fingerprint"] = canonical_hash(context)
    return context


def sample_observation_state(
    runtime: SimulatorRuntime,
    protocol: dict[str, Any],
    context: dict[str, Any],
    legitimate_seed: int,
) -> dict[str, Any]:
    """Sample normal-state variation while keeping the target scene fixed.

    The v1 protocol has no ``observation_variation`` section, so it returns the
    original fixed context and simulator defaults without consuming any extra
    random state.  v2 uses a separate RNG stream so paired normal/anomaly
    observations with the same legitimate seed receive exactly the same
    normal communication state.
    """

    variation = protocol.get("observation_variation")
    if not variation:
        return {
            "legitimate_frequency_bands": [
                dict(band) for band in context["legitimate_frequency_bands"]
            ],
            "legitimate_tx_power_dbm": float(runtime.config.tx_power),
            "noise_power_dbm": float(runtime.config.p_noise_dbm),
            "receiver_gain_db": 0.0,
            "state_seed": int(legitimate_seed),
        }

    rng = np.random.default_rng(
        np.random.SeedSequence([int(legitimate_seed), 0x4F46444D])
    )
    base_bands = [
        dict(band) for band in context["legitimate_frequency_bands"]
    ]
    max_shift = int(variation.get("frequency_band_shift_rb", 0))
    shift = 0
    if max_shift > 0 and base_bands:
        minimum_start = min(int(band["start_rb"]) for band in base_bands)
        maximum_end = max(int(band["end_rb"]) for band in base_bands)
        lower = max(-max_shift, -minimum_start)
        upper = min(max_shift, int(runtime.config.num_rb) - maximum_end)
        if lower <= upper:
            shift = int(rng.integers(lower, upper + 1))
    bands = [
        {
            "start_rb": int(band["start_rb"]) + shift,
            "end_rb": int(band["end_rb"]) + shift,
        }
        for band in base_bands
    ]

    tx_jitter = float(variation.get("tx_power_jitter_db", 0.0))
    noise_jitter = float(variation.get("noise_power_jitter_db", 0.0))
    receiver_jitter = float(variation.get("receiver_gain_jitter_db", 0.0))
    return {
        "legitimate_frequency_bands": bands,
        "frequency_band_shift_rb": shift,
        "legitimate_tx_power_dbm": float(
            runtime.config.tx_power
            + rng.uniform(-tx_jitter, tx_jitter)
            if tx_jitter > 0
            else runtime.config.tx_power
        ),
        "noise_power_dbm": float(
            runtime.config.p_noise_dbm
            + rng.uniform(-noise_jitter, noise_jitter)
            if noise_jitter > 0
            else runtime.config.p_noise_dbm
        ),
        "receiver_gain_db": float(
            rng.uniform(-receiver_jitter, receiver_jitter)
            if receiver_jitter > 0
            else 0.0
        ),
        "state_seed": int(legitimate_seed),
    }


def resources_for_observation(
    runtime: SimulatorRuntime,
    protocol: dict[str, Any],
    context: dict[str, Any],
    legitimate_seed: int,
    observation_state: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    if protocol.get("observation_variation"):
        rng = np.random.default_rng(
            np.random.SeedSequence([int(legitimate_seed), 0x524553])
        )
    else:
        rng = np.random.default_rng(legitimate_seed)
    minimum = float(protocol["normal_context"]["slot_utilization_min"])
    maximum = float(protocol["normal_context"]["slot_utilization_max"])
    bands = (
        observation_state["legitimate_frequency_bands"]
        if observation_state is not None
        else context["legitimate_frequency_bands"]
    )
    resources: list[np.ndarray] = []
    for band in bands:
        matrix = np.zeros(
            (runtime.config.num_rb, runtime.config.n_slots), dtype=bool
        )
        utilization = float(rng.uniform(minimum, maximum))
        active_slots = np.flatnonzero(
            rng.binomial(1, utilization, runtime.config.n_slots)
        )
        if len(active_slots) == 0:
            active_slots = np.asarray(
                [int(rng.integers(0, runtime.config.n_slots))], dtype=int
            )
        matrix[
            int(band["start_rb"]) : int(band["end_rb"]),
            active_slots,
        ] = True
        resources.append(matrix)
    return resources


def trace_legitimate_channel(
    runtime: SimulatorRuntime, context: dict[str, Any]
) -> np.ndarray:
    for tx_index, position in enumerate(
        context["legitimate_transmitter_positions"]
    ):
        runtime.scene.add(
            runtime.srt.Transmitter(
                name=f"tx{tx_index}", position=np.asarray(position)
            )
        )
    runtime.scene = runtime.rt_utils.configure_tx_antenna_pattern(
        runtime.scene, "iso"
    )
    try:
        paths = runtime.path_solver(
            scene=runtime.scene,
            max_depth=runtime.config.sionna.max_depth,
            samples_per_src=int(runtime.config.sionna.num_rays),
            specular_reflection=True,
            refraction=True,
        )
        channel = paths.cfr(
            runtime.fft_frequencies,
            sampling_frequency=runtime.config.subcarrier_spacing
            * runtime.config.nfft,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
    finally:
        for tx_index in range(
            len(context["legitimate_transmitter_positions"])
        ):
            runtime.scene.remove(f"tx{tx_index}")
    return channel


def sample_and_trace_jammer(
    runtime: SimulatorRuntime,
    jammer_type: str,
    jammer_seed: int,
    protocol: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    np.random.seed(jammer_seed)
    position = runtime.rt_utils.get_random_tx_location(
        runtime.config.scene_nr,
        runtime.scene_size,
        tx_height=runtime.config.sionna.tx_height,
    )
    orientation = [float(np.random.uniform(0, 2 * np.pi)), 0.0, 0.0]
    power_cfg = (
        protocol.get("jammer_power_dbm")
        if protocol is not None
        else None
    )
    if power_cfg:
        power_choices = np.arange(
            float(power_cfg["min"]),
            float(power_cfg["max"]) + 1.0,
            float(power_cfg["step"]),
        )
    else:
        power_choices = np.arange(
            runtime.config.jam_power.min,
            runtime.config.jam_power.max + 1,
            runtime.config.jam_power.step,
        )
    power = float(np.random.choice(power_choices))
    metadata = {
        "jammer_type": jammer_type,
        "jammer_power": power,
        "jammer_location": [float(value) for value in position],
        "jammer_orientation": orientation,
        "jammer_seed": jammer_seed,
    }

    runtime.scene = runtime.rt_utils.configure_tx_antenna_pattern(
        runtime.scene, runtime.jammer_pattern
    )
    runtime.scene.add(
        runtime.srt.Transmitter(name="jam0", position=np.asarray(position))
    )
    try:
        paths = runtime.path_solver(
            scene=runtime.scene,
            max_depth=runtime.config.sionna.max_depth,
            samples_per_src=int(runtime.config.sionna.num_rays),
            specular_reflection=True,
            refraction=True,
        )
        channel = paths.cfr(
            runtime.fft_frequencies,
            sampling_frequency=runtime.config.subcarrier_spacing
            * runtime.config.nfft,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
    finally:
        runtime.scene.remove("jam0")
    return metadata, channel


def build_sample(
    runtime: SimulatorRuntime,
    context: dict[str, Any],
    resources: list[np.ndarray],
    jammer: dict[str, Any] | None,
) -> Any:
    sample = runtime.Sample()
    for position, allocation in zip(
        context["legitimate_transmitter_positions"], resources
    ):
        sample.add_transmitter(
            runtime.Transmitter(position, np.array(allocation, copy=True))
        )
    if jammer is not None:
        sample.add_jammer(
            runtime.Jammer(
                jammer["jammer_location"],
                jammer["jammer_orientation"],
                jammer["jammer_power"],
                jammer["jammer_type"],
                runtime.jammer_pattern,
            )
        )
    return sample


def observation_metadata_path(observation_dir: Path) -> Path:
    return observation_dir / "observation.json"


def complete_observation(observation_dir: Path, expected_sus: int) -> bool:
    metadata_path = observation_metadata_path(observation_dir)
    if not metadata_path.is_file():
        return False
    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False
    images = metadata.get("images", [])
    return len(images) == expected_sus and all(
        (observation_dir / item["filename"]).is_file() for item in images
    )


def normalized_image(
    spectrogram: np.ndarray, minimum_db: float, maximum_db: float
) -> np.ndarray:
    values = np.nan_to_num(
        np.asarray(spectrogram, dtype=np.float32),
        nan=minimum_db,
        neginf=minimum_db,
        posinf=maximum_db,
    )
    values = np.clip(values, minimum_db, maximum_db)
    values = np.rint(
        (values - minimum_db) / (maximum_db - minimum_db) * 255.0
    )
    return values.astype(np.uint8)


def save_observation(
    runtime: SimulatorRuntime,
    protocol: dict[str, Any],
    context: dict[str, Any],
    legitimate_channel: np.ndarray,
    observation: dict[str, Any],
    scene_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    observation_dir = scene_dir / "images" / observation["observation_id"]
    expected_sus = int(protocol["sensing_units"])
    if resume and complete_observation(observation_dir, expected_sus):
        return read_json(observation_metadata_path(observation_dir))
    if observation_dir.exists():
        raise FileExistsError(
            f"Incomplete observation exists; rerun with a clean output or remove only "
            f"this incomplete directory: {observation_dir}"
        )

    temporary = observation_dir.with_name(
        f".{observation_dir.name}.tmp-{os.getpid()}"
    )
    for stale in observation_dir.parent.glob(f".{observation_dir.name}.tmp-*"):
        if stale != temporary:
            shutil.rmtree(stale)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    observation_state = sample_observation_state(
        runtime,
        protocol,
        context,
        int(observation["legitimate_seed"]),
    )
    resources = resources_for_observation(
        runtime,
        protocol,
        context,
        int(observation["legitimate_seed"]),
        observation_state=observation_state,
    )
    resource_fingerprint = array_hash(resources)

    jammer = None
    channel = legitimate_channel
    if observation["label"] == 1:
        jammer, jammer_channel = sample_and_trace_jammer(
            runtime,
            observation["jammer_type"],
            int(observation["jammer_seed"]),
            protocol=protocol,
        )
        channel = np.concatenate((legitimate_channel, jammer_channel), axis=2)

    sample = build_sample(runtime, context, resources, jammer)
    original_tx_power = runtime.config.tx_power
    original_noise_power = runtime.config.p_noise_dbm
    runtime.config.tx_power = observation_state["legitimate_tx_power_dbm"]
    runtime.config.p_noise_dbm = observation_state["noise_power_dbm"]
    np.random.seed(int(observation["legitimate_seed"]))
    try:
        sample = runtime.rt_utils.create_spectrograms(
            sample, runtime.config, channel, noise=True
        )
    finally:
        runtime.config.tx_power = original_tx_power
        runtime.config.p_noise_dbm = original_noise_power

    receiver_gain_db = float(observation_state.get("receiver_gain_db", 0.0))
    if abs(receiver_gain_db) > 1e-12:
        for su_id in sample.spectrograms:
            sample.spectrograms[su_id] = (
                np.asarray(sample.spectrograms[su_id]) + receiver_gain_db
            )

    normalization = protocol["normalization"]
    minimum_db = float(normalization["min_db"])
    maximum_db = float(normalization["max_db"])
    images: list[dict[str, Any]] = []
    for su_id in sorted(sample.spectrograms):
        image = normalized_image(
            sample.spectrograms[su_id], minimum_db, maximum_db
        )
        filename = f"spectrogram-su{int(su_id):02d}.png"
        Image.fromarray(image, mode="L").save(temporary / filename)
        images.append(
            {
                "su_id": int(su_id),
                "filename": filename,
                "array_shape": list(image.shape),
                "pixel_min": int(image.min()),
                "pixel_max": int(image.max()),
            }
        )

    metadata = dict(observation)
    metadata.update(
        {
            "target_scene_id": context["target_scene_id"],
            "split": context["split"],
            "context_seed": context["context_seed"],
            "context_fingerprint": context["context_fingerprint"],
            "resource_fingerprint": resource_fingerprint,
            "legitimate_frequency_bands": observation_state[
                "legitimate_frequency_bands"
            ],
            "frequency_band_shift_rb": observation_state.get(
                "frequency_band_shift_rb", 0
            ),
            "legitimate_tx_power_dbm": observation_state[
                "legitimate_tx_power_dbm"
            ],
            "noise_power_dbm": observation_state["noise_power_dbm"],
            "receiver_gain_db": receiver_gain_db,
            "num_legitimate_transmitters": context[
                "num_legitimate_transmitters"
            ],
            "jammer_power": None if jammer is None else jammer["jammer_power"],
            "jammer_location": None
            if jammer is None
            else jammer["jammer_location"],
            "jammer_orientation": None
            if jammer is None
            else jammer["jammer_orientation"],
            "images": images,
        }
    )
    write_json_atomic(temporary / "observation.json", metadata)
    observation_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, observation_dir)
    return metadata


def observation_plan(
    protocol: dict[str, Any],
    split: str,
    scene_index: int,
    effective_counts: dict[str, int],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index in range(effective_counts["normal_support"]):
        observations.append(
            {
                "observation_id": f"support_{index:03d}",
                "role": "normal_support",
                "support_rank": index + 1,
                "paired_normal_observation_id": None,
                "label": 0,
                "jammer_type": "no jammer",
                "jammer_seed": None,
                "legitimate_seed": derived_seed(
                    protocol["base_seed"], split, scene_index, stream=10, item=index
                ),
            }
        )

    normal_seeds: list[int] = []
    for index in range(effective_counts["normal_test"]):
        legitimate_seed = derived_seed(
            protocol["base_seed"], split, scene_index, stream=20, item=index
        )
        normal_seeds.append(legitimate_seed)
        observations.append(
            {
                "observation_id": f"normal_{index:03d}",
                "role": "normal_test",
                "support_rank": None,
                "paired_normal_observation_id": None,
                "label": 0,
                "jammer_type": "no jammer",
                "jammer_seed": None,
                "legitimate_seed": legitimate_seed,
            }
        )

    anomaly_index = 0
    for jammer_type in protocol["jammer_types"]:
        for type_index in range(effective_counts["anomaly_test_per_type"]):
            paired_normal_index = anomaly_index % len(normal_seeds)
            observations.append(
                {
                    "observation_id": f"{jammer_type}_{type_index:03d}",
                    "role": "anomaly_test",
                    "support_rank": None,
                    "paired_normal_observation_id": (
                        f"normal_{paired_normal_index:03d}"
                    ),
                    "label": 1,
                    "jammer_type": jammer_type,
                    "jammer_seed": derived_seed(
                        protocol["base_seed"],
                        split,
                        scene_index,
                        stream=30,
                        item=anomaly_index,
                    ),
                    "legitimate_seed": normal_seeds[paired_normal_index],
                }
            )
            anomaly_index += 1
    return observations


def manifest_rows(
    output_root: Path,
    metadata_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata in metadata_items:
        observation_base = (
            Path("scenes")
            / metadata["split"]
            / metadata["target_scene_id"]
            / "images"
            / metadata["observation_id"]
        )
        for image in metadata["images"]:
            rows.append(
                {
                    "target_scene_id": metadata["target_scene_id"],
                    "split": metadata["split"],
                    "observation_id": metadata["observation_id"],
                    "role": metadata["role"],
                    "support_rank": ""
                    if metadata["support_rank"] is None
                    else metadata["support_rank"],
                    "paired_normal_observation_id": metadata[
                        "paired_normal_observation_id"
                    ]
                    or "",
                    "label": metadata["label"],
                    "jammer_type": metadata["jammer_type"],
                    "jammer_power": ""
                    if metadata["jammer_power"] is None
                    else metadata["jammer_power"],
                    "jammer_location": ""
                    if metadata["jammer_location"] is None
                    else json.dumps(metadata["jammer_location"]),
                    "su_id": image["su_id"],
                    "image_path": str(observation_base / image["filename"]),
                    "context_seed": metadata["context_seed"],
                    "legitimate_seed": metadata["legitimate_seed"],
                    "jammer_seed": ""
                    if metadata["jammer_seed"] is None
                    else metadata["jammer_seed"],
                    "context_fingerprint": metadata["context_fingerprint"],
                    "resource_fingerprint": metadata["resource_fingerprint"],
                    "legitimate_frequency_bands": json.dumps(
                        metadata.get("legitimate_frequency_bands", [])
                    ),
                    "frequency_band_shift_rb": metadata.get(
                        "frequency_band_shift_rb", 0
                    ),
                    "legitimate_tx_power_dbm": metadata.get(
                        "legitimate_tx_power_dbm", ""
                    ),
                    "noise_power_dbm": metadata.get("noise_power_dbm", ""),
                    "receiver_gain_db": metadata.get("receiver_gain_db", 0.0),
                    "num_legitimate_transmitters": metadata[
                        "num_legitimate_transmitters"
                    ],
                }
            )
    for row in rows:
        if not (output_root / row["image_path"]).is_file():
            raise FileNotFoundError(output_root / row["image_path"])
    return rows


def generate_target_scene(
    runtime: SimulatorRuntime,
    protocol: dict[str, Any],
    output_root: Path,
    split: str,
    scene_index: int,
    effective_counts: dict[str, int],
    resume: bool,
) -> dict[str, Any]:
    context = sample_target_context(runtime, protocol, split, scene_index)
    scene_dir = (
        output_root / "scenes" / split / context["target_scene_id"]
    )
    completion_path = scene_dir / "_COMPLETE.json"
    if resume and completion_path.is_file():
        return read_json(completion_path)
    scene_dir.mkdir(parents=True, exist_ok=True)

    context_path = scene_dir / "context.json"
    if context_path.exists():
        stored = read_json(context_path)
        if stored != context:
            raise RuntimeError(
                f"Context mismatch for resumable scene: {context_path}"
            )
    else:
        write_json_atomic(context_path, context)

    started = time.time()
    legitimate_channel = trace_legitimate_channel(runtime, context)
    plan = observation_plan(protocol, split, scene_index, effective_counts)
    metadata_items: list[dict[str, Any]] = []
    for observation_index, observation in enumerate(plan, start=1):
        item_started = time.time()
        metadata = save_observation(
            runtime,
            protocol,
            context,
            legitimate_channel,
            observation,
            scene_dir,
            resume=resume,
        )
        metadata_items.append(metadata)
        print(
            json.dumps(
                {
                    "target_scene_id": context["target_scene_id"],
                    "observation": observation["observation_id"],
                    "progress": f"{observation_index}/{len(plan)}",
                    "seconds": round(time.time() - item_started, 3),
                }
            ),
            flush=True,
        )

    rows = manifest_rows(output_root, metadata_items)
    write_csv_atomic(scene_dir / "manifest.csv", rows)
    summary = {
        "target_scene_id": context["target_scene_id"],
        "split": split,
        "scene_index": scene_index,
        "context_fingerprint": context["context_fingerprint"],
        "observations": len(plan),
        "images": len(rows),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json_atomic(completion_path, summary)
    return summary


def effective_protocol(
    protocol: dict[str, Any],
    mode: str,
) -> tuple[dict[str, int], dict[str, int]]:
    if mode == "smoke":
        return (
            {"smoke": 1},
            {
                "normal_support": 4,
                "normal_test": 5,
                "anomaly_test_per_type": 1,
            },
        )
    return (
        {
            "validation": int(protocol["scene_counts"]["validation"]),
            "test": int(protocol["scene_counts"]["test"]),
        },
        {
            "normal_support": int(
                protocol["observations_per_scene"]["normal_support"]
            ),
            "normal_test": int(
                protocol["observations_per_scene"]["normal_test"]
            ),
            "anomaly_test_per_type": int(
                protocol["observations_per_scene"]["anomaly_test_per_type"]
            ),
        },
    )


def initialize_output(
    output_root: Path,
    protocol: dict[str, Any],
    mode: str,
    split_counts: dict[str, int],
    observation_counts: dict[str, int],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    effective = {
        "source_protocol": protocol,
        "mode": mode,
        "effective_scene_counts": split_counts,
        "effective_observations_per_scene": observation_counts,
        "generator": {
            "script": str(Path(__file__).relative_to(REPO_ROOT)),
            "python": sys.version.split()[0],
            "sionna": importlib.metadata.version("sionna"),
            "sionna_rt": importlib.metadata.version("sionna-rt"),
        },
    }
    protocol_path = output_root / "protocol.json"
    if protocol_path.exists():
        stored = read_json(protocol_path)
        if stored != effective:
            raise RuntimeError(
                f"Output root belongs to a different protocol: {protocol_path}"
            )
    else:
        write_json_atomic(protocol_path, effective)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate target-scene cold-start OFDMA spectrograms."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        help="Formal split to generate. Omit to generate both sequentially.",
    )
    parser.add_argument(
        "--scene-indexes",
        help="Comma-separated indexes or inclusive ranges, e.g. 0,2-4.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(args.config.resolve())
    default_output = Path(protocol["output_root"])
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (
            default_output.with_name(default_output.name + "-smoke")
            if args.mode == "smoke"
            else default_output
        )
    )
    split_counts, observation_counts = effective_protocol(protocol, args.mode)
    initialize_output(
        output_root, protocol, args.mode, split_counts, observation_counts
    )
    runtime = initialize_simulator(protocol)

    if args.mode == "smoke":
        requested = [("smoke", [0])]
    else:
        splits = [args.split] if args.split else ["validation", "test"]
        requested = [
            (
                split,
                parse_scene_indexes(args.scene_indexes, split_counts[split]),
            )
            for split in splits
        ]

    summaries: list[dict[str, Any]] = []
    for split, indexes in requested:
        for scene_index in indexes:
            summary = generate_target_scene(
                runtime,
                protocol,
                output_root,
                split,
                scene_index,
                observation_counts,
                resume=args.resume,
            )
            summaries.append(summary)
            print(json.dumps({"scene_complete": summary}), flush=True)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "completed_scenes": len(summaries),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
