from __future__ import annotations

import argparse
import json
import math
import queue
import re
import shutil
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
RENDERS_ROOT = WEB_ROOT / "assets" / "renders"
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PRESSURE_LEVELS = list(range(1000, 49, -50))
MAX_BODY_BYTES = 16 * 1024
MAX_PENDING_JOBS = 1
MAX_RETAINED_JOBS = 5


DEFAULT_PAYLOAD: dict[str, object] = {
    "scenario": "circulation",
    "resolution": 5.0,
    "level": 900,
    "region": "world",
    "season": "equinox",
    "spinupHours": 120.0,
    "analysisHours": 120.0,
    "frames": 48,
    "fps": 12,
    "particles": 2400,
    "flowSpeed": 1.0,
    "trail": 0.94,
    "tibetScale": 1.0,
    "landHeatingScale": 1.0,
    "oceanCurrentScale": 1.0,
    "jetStrength": 35.0,
    "perturbationAmplitude": 1.0,
    "hemisphere": "north",
    "equatorToPoleContrastK": 60.0,
    "surfaceDragDays": 1.0,
    "seasonalHeatEquatorDeg": 0.0,
}


CONFIG_RESPONSE: dict[str, object] = {
    "parameters": {
        "scenario": {
            "label": "实验场景",
            "type": "select",
            "options": [
                {"value": "circulation", "label": "全球地形环流 · HS + ETOPO + 闭合"},
                {"value": "baroclinic", "label": "干温带气旋 · 动力核验证"},
            ],
            "default": DEFAULT_PAYLOAD["scenario"],
        },
        "resolution": {
            "label": "模型分辨率",
            "type": "select",
            "options": [
                {"value": 1.0, "label": "1° · 高质量"},
                {"value": 2.5, "label": "2.5° · 平衡"},
                {"value": 5.0, "label": "5° · 快速"},
            ],
            "default": DEFAULT_PAYLOAD["resolution"],
        },
        "level": {
            "label": "气压层",
            "type": "select",
            "options": [{"value": level, "label": f"{level} hPa"} for level in PRESSURE_LEVELS],
            "default": DEFAULT_PAYLOAD["level"],
        },
        "season": {
            "label": "季节",
            "type": "select",
            "options": [
                {"value": "summer", "label": "北半球夏季"},
                {"value": "equinox", "label": "春秋分"},
                {"value": "winter", "label": "北半球冬季"},
            ],
            "default": DEFAULT_PAYLOAD["season"],
        },
        "spinupHours": {"label": "环流建立期", "type": "number", "min": 48.0, "max": 720.0, "step": 24.0, "unit": "h", "default": DEFAULT_PAYLOAD["spinupHours"]},
        "analysisHours": {"label": "播放分析窗", "type": "number", "min": 24.0, "max": 720.0, "step": 24.0, "unit": "h", "default": DEFAULT_PAYLOAD["analysisHours"]},
        "frames": {"label": "动画帧数", "type": "integer", "min": 24, "max": 120, "step": 1, "default": DEFAULT_PAYLOAD["frames"]},
        "fps": {"label": "播放帧率", "type": "integer", "min": 12, "max": 30, "step": 1, "unit": "fps", "default": DEFAULT_PAYLOAD["fps"]},
        "particles": {"label": "粒子数量", "type": "integer", "min": 800, "max": 6000, "step": 100, "default": DEFAULT_PAYLOAD["particles"]},
        "flowSpeed": {"label": "流线速度", "type": "number", "min": 0.25, "max": 3.0, "step": 0.05, "default": 1.0},
        "trail": {"label": "尾迹长度", "type": "number", "min": 0.75, "max": 0.985, "step": 0.005, "default": 0.94},
        "tibetScale": {"label": "青藏高原高度", "type": "number", "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0},
        "landHeatingScale": {"label": "陆地增温", "type": "number", "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0},
        "oceanCurrentScale": {"label": "洋流温度异常", "type": "number", "min": 0.0, "max": 2.0, "step": 0.05, "default": 1.0},
        "equatorToPoleContrastK": {"label": "赤道—极地温差", "type": "number", "min": 30.0, "max": 90.0, "step": 2.0, "unit": "K", "default": 60.0},
        "surfaceDragDays": {"label": "近地面摩擦时间", "type": "number", "min": 0.25, "max": 4.0, "step": 0.25, "unit": "d", "default": 1.0},
        "seasonalHeatEquatorDeg": {"label": "热赤道纬度", "type": "number", "min": -23.5, "max": 23.5, "step": 0.5, "unit": "°", "default": 0.0},
        "jetStrength": {"label": "急流峰值", "type": "number", "min": 25.0, "max": 45.0, "step": 1.0, "unit": "m/s", "default": 35.0},
        "perturbationAmplitude": {"label": "触发扰动", "type": "number", "min": 0.0, "max": 2.0, "step": 0.1, "unit": "m/s", "default": 1.0},
        "hemisphere": {
            "label": "发展半球",
            "type": "select",
            "options": [
                {"value": "north", "label": "北半球"},
                {"value": "south", "label": "南半球"},
            ],
            "default": "north",
        },
    },
    "presets": [
        {
            "id": "preview",
            "label": "快速三圈",
            "settings": {**DEFAULT_PAYLOAD, "resolution": 5.0, "spinupHours": 72.0, "analysisHours": 72.0, "frames": 30, "fps": 12, "particles": 1600},
        },
        {"id": "balanced", "label": "标准三圈", "settings": dict(DEFAULT_PAYLOAD)},
        {
            "id": "quality",
            "label": "2.5°三圈",
            "settings": {**DEFAULT_PAYLOAD, "resolution": 2.5, "spinupHours": 168.0, "analysisHours": 168.0, "frames": 60, "fps": 12, "particles": 3600},
        },
    ],
    "limits": {
        "maxPendingJobs": MAX_PENDING_JOBS,
        "retainedRenders": MAX_RETAINED_JOBS,
        "loopSeconds": {"min": 1.0, "max": 6.0},
        "particleFrameBudget": 750_000,
    },
    "estimateSecondsByResolution": {
        "1": 24_000,
        "2.5": 3_000,
        "5": 360,
    },
    "estimateNote": "按5天环流建立和5天连续分析窗估算；1°全球环流需要数小时，建议先用5°。",
}


class ValidationError(ValueError):
    pass


class RenderQueueFull(RuntimeError):
    pass


class RenderBusy(RuntimeError):
    pass


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(payload: dict[str, object], key: str, minimum: float, maximum: float) -> float:
    value = payload[key]
    if not _is_number(value):
        raise ValidationError(f"{key} 必须是有限数值")
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise ValidationError(f"{key} 必须在 {minimum:g}–{maximum:g} 之间")
    return parsed


def _integer(payload: dict[str, object], key: str, minimum: int, maximum: int) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{key} 必须是整数")
    if not minimum <= value <= maximum:
        raise ValidationError(f"{key} 必须在 {minimum}–{maximum} 之间")
    return value


def validate_render_payload(raw_payload: object):
    """Validate an API payload and return a renderer settings object."""

    if not isinstance(raw_payload, dict):
        raise ValidationError("请求正文必须是 JSON 对象")
    unknown = sorted(set(raw_payload) - set(DEFAULT_PAYLOAD))
    if unknown:
        raise ValidationError(f"不支持的参数：{', '.join(unknown)}")
    payload = {**DEFAULT_PAYLOAD, **raw_payload}

    scenario = payload["scenario"]
    if not isinstance(scenario, str) or scenario not in {"baroclinic", "circulation"}:
        raise ValidationError("scenario 仅支持 baroclinic 或 circulation")

    resolution = _number(payload, "resolution", 1.0, 5.0)
    if resolution not in {1.0, 2.5, 5.0}:
        raise ValidationError("resolution 仅支持 1、2.5 或 5")
    level = _integer(payload, "level", 50, 1000)
    if level not in PRESSURE_LEVELS:
        raise ValidationError("level 必须是 50 hPa 间隔的标准气压层")
    if scenario == "baroclinic" and level > 900:
        raise ValidationError("干温带气旋实验仅支持 900–50 hPa；建议查看 850 hPa 锋面")
    region = payload["region"]
    if not isinstance(region, str):
        raise ValidationError("region 必须是字符串")
    legacy_regions = {"east_asia", "east_asia_pacific", "asia", "global", "world"}
    if region not in legacy_regions:
        raise ValidationError("region 仅支持 world；旧版地图范围会自动归一化为 world")
    region = "world"
    season = payload["season"]
    if not isinstance(season, str) or season not in {"summer", "equinox", "winter"}:
        raise ValidationError("season 仅支持 summer、equinox 或 winter")
    hemisphere = payload["hemisphere"]
    if not isinstance(hemisphere, str) or hemisphere not in {"north", "south"}:
        raise ValidationError("hemisphere 仅支持 north 或 south")

    frames = _integer(payload, "frames", 24, 120)
    fps = _integer(payload, "fps", 12, 30)
    particles = _integer(payload, "particles", 800, 6000)
    if not 1.0 <= frames / fps <= 6.0:
        raise ValidationError("frames / fps 得到的循环时长必须在 1–6 秒之间")
    if (frames + 24) * particles > 750_000:
        raise ValidationError("帧数与粒子数量的组合过大，请降低其中一项")

    from scripts.prerender_windy import RenderSettings

    spinup_hours = _number(payload, "spinupHours", 0.0, 720.0)
    if scenario == "baroclinic" and not 72.0 <= spinup_hours <= 240.0:
        raise ValidationError("干温带气旋实验需要积分 72–240 小时")
    if scenario == "circulation" and not 48.0 <= spinup_hours <= 720.0:
        raise ValidationError("全球三圈环流需要 2–30 天建立期")

    analysis_hours = _number(payload, "analysisHours", 24.0, 720.0)

    return RenderSettings(
        scenario=scenario,
        resolution=resolution,
        level=level,
        region=region,
        season=season,
        spinup_hours=spinup_hours,
        analysis_hours=analysis_hours,
        frames=frames,
        fps=fps,
        particles=particles,
        flow_speed=_number(payload, "flowSpeed", 0.25, 3.0),
        trail=_number(payload, "trail", 0.75, 0.985),
        tibet_scale=_number(payload, "tibetScale", 0.0, 2.0),
        land_heating_scale=_number(payload, "landHeatingScale", 0.0, 2.0),
        ocean_current_scale=_number(payload, "oceanCurrentScale", 0.0, 2.0),
        jet_strength=_number(payload, "jetStrength", 25.0, 45.0),
        perturbation_amplitude=_number(payload, "perturbationAmplitude", 0.0, 2.0),
        hemisphere=hemisphere,
        equator_to_pole_contrast_k=_number(payload, "equatorToPoleContrastK", 30.0, 90.0),
        surface_drag_days=_number(payload, "surfaceDragDays", 0.25, 4.0),
        seasonal_heat_equator_deg=_number(payload, "seasonalHeatEquatorDeg", -23.5, 23.5),
    )


def _public_settings(settings: object) -> dict[str, object]:
    values = asdict(settings)
    return {
        "scenario": values["scenario"],
        "resolution": values["resolution"],
        "level": values["level"],
        "region": values["region"],
        "season": values["season"],
        "spinupHours": values["spinup_hours"],
        "analysisHours": values["analysis_hours"],
        "frames": values["frames"],
        "fps": values["fps"],
        "particles": values["particles"],
        "flowSpeed": values["flow_speed"],
        "trail": values["trail"],
        "tibetScale": values["tibet_scale"],
        "landHeatingScale": values["land_heating_scale"],
        "oceanCurrentScale": values["ocean_current_scale"],
        "jetStrength": values["jet_strength"],
        "perturbationAmplitude": values["perturbation_amplitude"],
        "hemisphere": values["hemisphere"],
        "equatorToPoleContrastK": values["equator_to_pole_contrast_k"],
        "surfaceDragDays": values["surface_drag_days"],
        "seasonalHeatEquatorDeg": values["seasonal_heat_equator_deg"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class RenderJob:
    job_id: str
    settings: object
    status: str = "queued"
    progress: float = 0.0
    stage: str = "queued"
    message: str = "已进入渲染队列"
    manifest_url: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    sequence: int = 0

    def public(self) -> dict[str, object]:
        result: dict[str, object] = {
            "jobId": self.job_id,
            "status": self.status,
            "progress": round(min(1.0, max(0.0, self.progress)), 4),
            "stage": self.stage,
            "message": self.message,
            "statusUrl": f"/api/render/{self.job_id}",
            "settings": _public_settings(self.settings),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if self.status == "complete" and self.manifest_url:
            result["manifestUrl"] = self.manifest_url
        return result


class RenderJobManager:
    """Serialize expensive model runs and publish only rendered media."""

    def __init__(self, renders_root: Path) -> None:
        self.renders_root = renders_root.resolve()
        self.renders_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, RenderJob] = {}
        self._sequence = 0
        self._queue: queue.Queue[tuple[str, object] | None] = queue.Queue(maxsize=MAX_PENDING_JOBS)
        self._stop = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, name="atmos20-render-worker", daemon=True)
        self._worker_thread.start()

    def submit(self, settings: object) -> RenderJob:
        with self._lock:
            if any(job.status in {"queued", "running"} for job in self._jobs.values()):
                raise RenderBusy("已有渲染任务正在计算，请等待它完成")
            self._sequence += 1
            job = RenderJob(job_id=uuid.uuid4().hex, settings=settings, sequence=self._sequence)
            self._jobs[job.job_id] = job
            try:
                self._queue.put_nowait((job.job_id, settings))
            except queue.Full as exc:
                self._jobs.pop(job.job_id, None)
                raise RenderQueueFull("渲染队列已满，请等待当前任务完成") from exc
            return job

    def get(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.public() if job else None

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _update(self, job_id: str, **updates: object) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = _utc_now()

    def _worker(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            try:
                if item is None or self._stop.is_set():
                    return
                job_id, settings = item
                self._run_job(job_id, settings)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str, settings: object) -> None:
        from scripts.prerender_windy import render_assets

        output = (self.renders_root / job_id).resolve()
        if output.parent != self.renders_root or not JOB_ID_RE.fullmatch(job_id):
            self._update(job_id, status="error", stage="error", message="内部输出路径校验失败")
            return
        self._update(job_id, status="running", progress=0.0, stage="initializing", message="正在启动后端计算")

        def report(progress: float, stage: str, message: str) -> None:
            self._update(job_id, progress=float(progress), stage=stage, message=message)

        try:
            manifest = render_assets(
                settings,
                output,
                asset_base=f"/assets/renders/{job_id}/",
                progress=report,
            )
            if (
                getattr(settings, "scenario", None) == "circulation"
                and manifest.get("qualityGatePassed") is not True
            ):
                self._safe_remove(output)
                self._update(
                    job_id,
                    status="error",
                    progress=1.0,
                    stage="error",
                    message="计算完成，但五风带或高波数数值门禁未通过；结果未发布",
                )
                self._prune()
                return
            manifest_url = f"/assets/renders/{job_id}/manifest.json"
            self._update(
                job_id,
                status="complete",
                progress=1.0,
                stage="complete",
                message="渲染完成，可以播放",
                manifest_url=manifest_url,
            )
            self._prune()
        except Exception:
            traceback.print_exc()
            self._safe_remove(output)
            self._update(
                job_id,
                status="error",
                progress=0.0,
                stage="error",
                message="渲染失败；请降低分辨率、帧数或粒子数量后重试",
            )
            self._prune()

    def _safe_remove(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved.parent != self.renders_root or not JOB_ID_RE.fullmatch(resolved.name):
            raise RuntimeError(f"Refusing to delete unsafe render path: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)

    def _prune(self) -> None:
        with self._lock:
            active_ids = {job.job_id for job in self._jobs.values() if job.status in {"queued", "running"}}
            directories = []
            for path in self.renders_root.iterdir():
                resolved = path.resolve()
                if (
                    path.is_dir()
                    and resolved.parent == self.renders_root
                    and JOB_ID_RE.fullmatch(path.name)
                    and path.name not in active_ids
                ):
                    directories.append(path)
            directories.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
            for stale in directories[MAX_RETAINED_JOBS:]:
                self._safe_remove(stale)

            terminal = sorted(
                (job for job in self._jobs.values() if job.status in {"complete", "error"}),
                key=lambda job: job.sequence,
                reverse=True,
            )
            for stale_job in terminal[MAX_RETAINED_JOBS:]:
                self._jobs.pop(stale_job.job_id, None)


JOB_MANAGER: RenderJobManager | None = None


class AtmosMapHandler(SimpleHTTPRequestHandler):
    """Serve the map UI plus the small render-control API."""

    server_version = "Atmos20/2"
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".webp": "image/webp",
        ".json": "application/json; charset=utf-8",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self) -> None:
        path = urlsplit(self.path).path
        if (
            path in {"", "/"}
            or path.endswith(".html")
            or path.endswith("manifest.json")
            or path.endswith((".css", ".js"))
        ):
            self.send_header("Cache-Control", "no-store")
        elif path.endswith(".webp"):
            self.send_header("Cache-Control", "public, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/api/config":
            self._send_json(HTTPStatus.OK, CONFIG_RESPONSE)
            return
        if path.startswith("/api/render/"):
            job_id = path.removeprefix("/api/render/")
            if not JOB_ID_RE.fullmatch(job_id):
                self._send_error_json(HTTPStatus.NOT_FOUND, "未找到该渲染任务")
                return
            job = JOB_MANAGER.get(job_id) if JOB_MANAGER else None
            if job is None:
                self._send_error_json(HTTPStatus.NOT_FOUND, "未找到该渲染任务")
                return
            self._send_json(HTTPStatus.OK, job)
            return
        if path.startswith("/api/"):
            self._send_error_json(HTTPStatus.NOT_FOUND, "API 路径不存在")
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        if path != "/api/render":
            self._send_error_json(HTTPStatus.NOT_FOUND, "API 路径不存在")
            return
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self._send_error_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type 必须是 application/json")
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "Content-Length 无效")
            return
        if content_length <= 0:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "请求正文不能为空")
            return
        if content_length > MAX_BODY_BYTES:
            self._send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求正文过大")
            return
        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
            settings = validate_render_payload(payload)
            if JOB_MANAGER is None:
                raise RuntimeError("渲染服务尚未启动")
            job = JOB_MANAGER.submit(settings)
        except UnicodeDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "请求正文必须使用 UTF-8")
            return
        except json.JSONDecodeError:
            self._send_error_json(HTTPStatus.BAD_REQUEST, "请求正文不是有效 JSON")
            return
        except ValidationError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except RenderBusy as exc:
            self._send_error_json(HTTPStatus.CONFLICT, str(exc))
            return
        except RenderQueueFull as exc:
            self._send_error_json(HTTPStatus.TOO_MANY_REQUESTS, str(exc))
            return
        except RuntimeError as exc:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        self._send_json(
            HTTPStatus.ACCEPTED,
            {"jobId": job.job_id, "statusUrl": f"/api/render/{job.job_id}"},
        )

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message, "status": int(status)})

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        super().log_message(format_string, *args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Atmos20 with on-demand backend rendering.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    manifest = WEB_ROOT / "assets" / "prerender" / "manifest.json"
    if not manifest.exists():
        raise SystemExit("Missing initial assets. Run: python scripts/prerender_windy.py")

    global JOB_MANAGER
    JOB_MANAGER = RenderJobManager(RENDERS_ROOT)
    server = ThreadingHTTPServer((args.host, args.port), AtmosMapHandler)
    print(f"Atmos20 map with backend rendering: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        JOB_MANAGER.shutdown()


if __name__ == "__main__":
    main()
