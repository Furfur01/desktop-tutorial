"""Build named Atmos20 pre-render bundles and their browser catalog.

A single atmospheric trajectory remains serial in model time. This command
parallelizes independent profiles, which is the safe and reproducible unit of
work for local multi-core machines and GitHub Actions matrix jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "prerender" / "profiles.json"
DEFAULT_OUTPUT_ROOT = ROOT / "web" / "assets" / "prerenders"
DEFAULT_CATALOG = DEFAULT_OUTPUT_ROOT / "catalog.json"
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,79}$")


class ProfileError(ValueError):
    """Raised when a pre-render profile is malformed or unavailable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if int(registry.get("schemaVersion", 0)) != 1:
        raise ProfileError(f"unsupported profile schema in {path}")
    defaults = registry.get("defaults", {})
    profiles = registry.get("profiles")
    if not isinstance(defaults, dict) or not isinstance(profiles, list):
        raise ProfileError("registry must contain object defaults and a profile list")

    indexed: dict[str, dict[str, Any]] = {}
    for raw in profiles:
        if not isinstance(raw, dict):
            raise ProfileError("every profile must be an object")
        profile_id = raw.get("id")
        if not isinstance(profile_id, str) or not PROFILE_ID_RE.fullmatch(profile_id):
            raise ProfileError(f"invalid profile id: {profile_id!r}")
        if profile_id in indexed:
            raise ProfileError(f"duplicate profile id: {profile_id}")
        settings = raw.get("settings")
        if not isinstance(settings, dict):
            raise ProfileError(f"{profile_id}: settings must be an object")
        merged = dict(raw)
        merged["settings"] = {**defaults, **settings}
        indexed[profile_id] = merged
    return {
        "schemaVersion": 1,
        "defaults": defaults,
        "profiles": indexed,
        "order": [item["id"] for item in profiles],
    }


def _days(value: Any, *, field: str, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{field} must be a number") from exc
    if not parsed >= minimum:
        raise ProfileError(f"{field} must be >= {minimum:g}")
    return parsed


def profile_to_settings(profile: dict[str, Any]):
    from scripts.prerender_windy import RenderSettings

    raw = profile["settings"]
    scenario = str(raw.get("scenario", "circulation"))
    if scenario not in {"circulation", "baroclinic"}:
        raise ProfileError(f"{profile['id']}: unsupported scenario {scenario!r}")

    spinup_days = _days(raw.get("spinupDays", 0), field="spinupDays")
    analysis_days = _days(raw.get("analysisDays", 0), field="analysisDays")
    if scenario == "circulation" and analysis_days <= 0:
        raise ProfileError(f"{profile['id']}: circulation requires analysisDays > 0")
    if scenario == "baroclinic" and not 3 <= spinup_days <= 20:
        raise ProfileError(f"{profile['id']}: baroclinic life cycle must be 3-20 days")

    seasonal_heat = raw.get("seasonalHeatEquatorDeg")
    seasonal_heat_value = None if seasonal_heat is None else float(seasonal_heat)
    return RenderSettings(
        scenario=scenario,
        resolution=float(raw.get("resolution", 5.0)),
        level=int(raw.get("level", 900)),
        region=str(raw.get("region", "world")),
        season=str(raw.get("season", "equinox")),
        spinup_hours=spinup_days * 24.0,
        analysis_hours=analysis_days * 24.0,
        frames=int(raw.get("frames", 48)),
        fps=int(raw.get("fps", 12)),
        particles=int(raw.get("particles", 2400)),
        flow_speed=float(raw.get("flowSpeed", 1.0)),
        trail=float(raw.get("trail", 0.94)),
        tibet_scale=float(raw.get("tibetScale", 1.0)),
        land_heating_scale=float(raw.get("landHeatingScale", 1.0)),
        ocean_current_scale=float(raw.get("oceanCurrentScale", 1.0)),
        jet_strength=float(raw.get("jetStrength", 35.0)),
        perturbation_amplitude=float(raw.get("perturbationAmplitude", 1.0)),
        hemisphere=str(raw.get("hemisphere", "north")),
        equator_to_pole_contrast_k=float(raw.get("equatorToPoleContrastK", 60.0)),
        surface_drag_days=float(raw.get("surfaceDragDays", 1.0)),
        seasonal_heat_equator_deg=seasonal_heat_value,
    )


def _simulation_days(profile: dict[str, Any]) -> float:
    settings = profile["settings"]
    spinup = float(settings.get("spinupDays", 0))
    if settings.get("scenario", "circulation") == "baroclinic":
        return spinup
    return spinup + float(settings.get("analysisDays", 0))


def _profile_public_metadata(profile: dict[str, Any]) -> dict[str, Any]:
    cloud = profile.get("cloud") if isinstance(profile.get("cloud"), dict) else {}
    return {
        "id": profile["id"],
        "label": profile.get("label", {}),
        "description": profile.get("description", {}),
        "tags": profile.get("tags", []),
        "settings": profile["settings"],
        "simulationDays": _simulation_days(profile),
        "analysisDays": float(profile["settings"].get("analysisDays", 0)),
        "cloud": {
            "enabled": bool(cloud.get("enabled", False)),
            "estimatedRunnerMinutes": cloud.get("estimatedRunnerMinutes"),
        },
    }


def render_profile(
    profile_id: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    try:
        profile = registry["profiles"][profile_id]
    except KeyError as exc:
        raise ProfileError(f"unknown profile: {profile_id}") from exc

    from scripts.prerender_windy import render_assets

    settings = profile_to_settings(profile)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_output = (output_root / profile_id).resolve()
    if final_output.parent != output_root:
        raise ProfileError(f"unsafe output path for {profile_id}")

    temporary = (output_root / f".{profile_id}.{os.getpid()}.tmp").resolve()
    if temporary.parent != output_root:
        raise ProfileError(f"unsafe temporary path for {profile_id}")
    if temporary.exists():
        shutil.rmtree(temporary)

    print(
        f"[{profile_id}] scenario={settings.scenario} "
        f"grid={settings.resolution:g}° simulation={_simulation_days(profile):g} d "
        f"frames={settings.frames}",
        flush=True,
    )

    try:
        manifest = render_assets(
            settings,
            temporary,
            asset_base=f"/assets/prerenders/{profile_id}/",
            progress=lambda value, stage, message: print(
                f"[{profile_id}] {value:6.1%} [{stage}] {message}",
                flush=True,
            ),
            clean_output=True,
        )
        if (
            settings.scenario == "circulation"
            and manifest.get("qualityGatePassed") is not True
        ):
            raise RuntimeError(
                f"{profile_id}: circulation quality gate failed; bundle was not published"
            )

        manifest["precomputed"] = True
        manifest["profile"] = _profile_public_metadata(profile)
        manifest["renderExecution"] = {
            "cpuCount": os.cpu_count(),
            "python": sys.version.split()[0],
            "generatedBy": "scripts/prerender_profiles.py",
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if final_output.exists():
            shutil.rmtree(final_output)
        temporary.replace(final_output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    size_bytes = sum(
        path.stat().st_size for path in final_output.rglob("*") if path.is_file()
    )
    result = {
        "profile": profile_id,
        "output": str(final_output),
        "sizeMiB": round(size_bytes / (1024 * 1024), 2),
        "manifest": str(final_output / "manifest.json"),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _render_worker(arguments: tuple[str, str, str]) -> dict[str, Any]:
    profile_id, registry_path, output_root = arguments
    return render_profile(
        profile_id,
        registry_path=Path(registry_path),
        output_root=Path(output_root),
    )


def parse_selection(
    value: str,
    registry: dict[str, Any],
    *,
    hosted_only: bool = False,
) -> list[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        requested = list(registry["order"])
    unknown = [item for item in requested if item not in registry["profiles"]]
    if unknown:
        raise ProfileError(f"unknown profile(s): {', '.join(unknown)}")

    selected: list[str] = []
    for profile_id in requested:
        if profile_id in selected:
            continue
        profile = registry["profiles"][profile_id]
        cloud = profile.get("cloud") if isinstance(profile.get("cloud"), dict) else {}
        if hosted_only and not bool(cloud.get("enabled", False)):
            continue
        selected.append(profile_id)
    if not selected:
        raise ProfileError("profile selection is empty")
    return selected


def render_many(
    profile_ids: Iterable[str],
    *,
    registry_path: Path,
    output_root: Path,
    jobs: int,
) -> list[dict[str, Any]]:
    selected = list(profile_ids)
    if jobs <= 1 or len(selected) == 1:
        return [
            render_profile(
                profile_id,
                registry_path=registry_path,
                output_root=output_root,
            )
            for profile_id in selected
        ]

    worker_count = min(jobs, len(selected))
    results: list[dict[str, Any]] = []
    arguments = [
        (profile_id, str(registry_path.resolve()), str(output_root.resolve()))
        for profile_id in selected
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_render_worker, item): item[0] for item in arguments}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: selected.index(item["profile"]))
    return results


def build_catalog(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    output_root = output_root.resolve()
    entries: list[dict[str, Any]] = []

    for profile_id in registry["order"]:
        profile = registry["profiles"][profile_id]
        metadata = _profile_public_metadata(profile)
        manifest_path = output_root / profile_id / "manifest.json"
        manifest_url: str | None = None

        if manifest_path.is_file():
            manifest_url = f"/assets/prerenders/{profile_id}/manifest.json"
        else:
            legacy_path_value = profile.get("legacyManifestPath")
            legacy_url = profile.get("legacyManifestUrl")
            if isinstance(legacy_path_value, str) and isinstance(legacy_url, str):
                candidate = (ROOT / legacy_path_value).resolve()
                if candidate.is_file():
                    manifest_path = candidate
                    manifest_url = legacy_url

        manifest: dict[str, Any] = {}
        if manifest_url is not None:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ProfileError(
                    f"{profile_id}: cannot read generated manifest {manifest_path}"
                ) from exc

        entries.append(
            {
                **metadata,
                "available": manifest_url is not None,
                "manifestUrl": manifest_url,
                "generated": manifest.get("generated"),
                "frames": manifest.get(
                    "frames", int(profile["settings"].get("frames", 0))
                ),
                "fps": manifest.get("fps", int(profile["settings"].get("fps", 0))),
                "qualityGatePassed": manifest.get("qualityGatePassed"),
            }
        )

    catalog = {
        "schemaVersion": 1,
        "generated": _utc_now(),
        "profiles": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Wrote {output} with {sum(item['available'] for item in entries)} available profiles")
    return catalog


def _default_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(3, cpu_count // 2 or 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render named Atmos20 profiles and build a browser catalog."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available profile definitions.")
    list_parser.add_argument("--json", action="store_true", help="Print JSON metadata.")

    matrix_parser = subparsers.add_parser(
        "matrix", help="Emit a GitHub Actions-compatible profile matrix."
    )
    matrix_parser.add_argument("--select", default="all", help="Comma-separated IDs or all.")
    matrix_parser.add_argument("--hosted-only", action="store_true")
    matrix_parser.add_argument(
        "--github-output",
        type=Path,
        help="Append matrix=<json> to this GitHub output file.",
    )

    render_parser = subparsers.add_parser("render", help="Render one profile.")
    render_parser.add_argument("--profile", required=True)
    render_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    many_parser = subparsers.add_parser(
        "render-many", help="Render independent profiles in multiple processes."
    )
    many_parser.add_argument("--select", default="all", help="Comma-separated IDs or all.")
    many_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    many_parser.add_argument("--jobs", type=int, default=_default_jobs())

    catalog_parser = subparsers.add_parser(
        "catalog", help="Rebuild the browser pre-render catalog."
    )
    catalog_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    catalog_parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)

    args = parser.parse_args()
    registry = load_registry(args.registry)

    if args.command == "list":
        items = [
            _profile_public_metadata(registry["profiles"][profile_id])
            for profile_id in registry["order"]
        ]
        if args.json:
            print(json.dumps(items, ensure_ascii=False, indent=2))
        else:
            for item in items:
                label = item["label"].get("zh") or item["label"].get("en") or item["id"]
                print(
                    f"{item['id']:<38} {item['simulationDays']:>6g} d  "
                    f"{item['settings'].get('resolution', 5):>3g}°  {label}"
                )
        return

    if args.command == "matrix":
        selected = parse_selection(
            args.select, registry, hosted_only=args.hosted_only
        )
        matrix = json.dumps({"profile": selected}, separators=(",", ":"))
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write(f"matrix={matrix}\n")
        else:
            print(matrix)
        return

    if args.command == "render":
        render_profile(
            args.profile,
            registry_path=args.registry,
            output_root=args.output_root,
        )
        return

    if args.command == "render-many":
        if args.jobs < 1:
            raise SystemExit("--jobs must be at least 1")
        selected = parse_selection(args.select, registry)
        render_many(
            selected,
            registry_path=args.registry,
            output_root=args.output_root,
            jobs=args.jobs,
        )
        build_catalog(
            registry_path=args.registry,
            output_root=args.output_root,
            output=args.output_root / "catalog.json",
        )
        return

    if args.command == "catalog":
        build_catalog(
            registry_path=args.registry,
            output_root=args.output_root,
            output=args.output,
        )
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
