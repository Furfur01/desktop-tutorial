"""Validate an Atmos20 pre-render bundle, including real frame motion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


class ValidationError(RuntimeError):
    """Raised when a bundle is incomplete, static, or internally inconsistent."""


def _sample_hashes(image: Image.Image, frame_count: int) -> set[str]:
    sample_indexes = sorted(
        {
            0,
            max(0, frame_count // 4),
            max(0, frame_count // 2),
            max(0, (3 * frame_count) // 4),
            max(0, frame_count - 1),
        }
    )
    hashes: set[str] = set()
    for index in sample_indexes:
        image.seek(index)
        thumbnail = image.convert("RGB").resize((96, 48))
        hashes.add(hashlib.sha256(thumbnail.tobytes()).hexdigest())
    return hashes


def _validate_animation(
    path: Path,
    *,
    expected_frames: int,
    require_motion: bool,
    label: str,
) -> dict[str, object]:
    if not path.is_file():
        raise ValidationError(f"{label}: missing asset {path}")
    with Image.open(path) as image:
        actual_frames = int(getattr(image, "n_frames", 1))
        if actual_frames != expected_frames:
            raise ValidationError(
                f"{label}: expected {expected_frames} frames, found {actual_frames}"
            )
        unique_samples = len(_sample_hashes(image, actual_frames))
    if require_motion and expected_frames > 1 and unique_samples < 2:
        raise ValidationError(f"{label}: sampled frames are visually identical")
    return {
        "asset": path.name,
        "frames": actual_frames,
        "uniqueSampledFrames": unique_samples,
    }


def _iter_layer_assets(manifest: dict[str, object]) -> Iterable[tuple[str, str, bool]]:
    layers = manifest.get("layers")
    if not isinstance(layers, dict):
        raise ValidationError("manifest.layers must be an object")
    for layer_name, raw_layer in layers.items():
        if not isinstance(raw_layer, dict):
            raise ValidationError(f"layer {layer_name!r} must be an object")
        assets = raw_layer.get("assets")
        if not isinstance(assets, dict) or not assets:
            raise ValidationError(f"layer {layer_name!r} has no assets")
        animated = bool(raw_layer.get("animated", False))
        for level, filename in assets.items():
            if not isinstance(filename, str):
                raise ValidationError(
                    f"layer {layer_name!r} at {level} has a non-string asset"
                )
            yield f"layer {layer_name}@{level}", filename, animated


def validate_manifest(manifest_path: Path) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bundle_root = manifest_path.parent

    expected_frames = int(manifest.get("frames", 0))
    if expected_frames < 2:
        raise ValidationError("manifest.frames must be at least 2 for dynamic playback")
    fps = int(manifest.get("fps", 0))
    if fps < 1:
        raise ValidationError("manifest.fps must be positive")

    timeline = manifest.get("timeline")
    if not isinstance(timeline, list) or len(timeline) != expected_frames:
        raise ValidationError(
            f"timeline length must equal manifest.frames ({expected_frames})"
        )
    times: list[float] = []
    for item in timeline:
        if not isinstance(item, dict):
            raise ValidationError("timeline entries must be objects")
        raw_time = item.get("modelHour", item.get("forecastHour"))
        try:
            times.append(float(raw_time))
        except (TypeError, ValueError) as exc:
            raise ValidationError("timeline entries need modelHour or forecastHour") from exc
    if any(later < earlier for earlier, later in zip(times, times[1:])):
        raise ValidationError("timeline must be monotonic")
    if times[-1] <= times[0]:
        raise ValidationError("timeline has no simulated-time motion")

    scenario = (
        manifest.get("settings", {}).get("scenario")
        if isinstance(manifest.get("settings"), dict)
        else None
    )
    if scenario == "circulation" and manifest.get("qualityGatePassed") is not True:
        raise ValidationError("circulation quality gate did not pass")

    checks: list[dict[str, object]] = []
    for label, filename, animated in _iter_layer_assets(manifest):
        expected = expected_frames if animated else 1
        checks.append(
            _validate_animation(
                bundle_root / filename,
                expected_frames=expected,
                require_motion=animated,
                label=label,
            )
        )

    particles = manifest.get("particles")
    if not isinstance(particles, dict) or not particles:
        raise ValidationError("manifest.particles must be a non-empty object")
    for level, filename in particles.items():
        if not isinstance(filename, str):
            raise ValidationError(f"particle asset at {level} is not a string")
        checks.append(
            _validate_animation(
                bundle_root / filename,
                expected_frames=expected_frames,
                require_motion=True,
                label=f"particles@{level}",
            )
        )

    summary = {
        "manifest": str(manifest_path),
        "scenario": scenario,
        "frames": expected_frames,
        "fps": fps,
        "simulatedHours": round(times[-1] - times[0], 3),
        "checkedAssets": len(checks),
        "assets": checks,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate frame counts, timeline consistency and visible motion."
    )
    parser.add_argument("manifest", type=Path, nargs="+")
    args = parser.parse_args()

    failures: list[str] = []
    for manifest_path in args.manifest:
        try:
            validate_manifest(manifest_path)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            failures.append(f"{manifest_path}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
