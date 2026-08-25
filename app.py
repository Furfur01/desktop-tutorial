from __future__ import annotations

import threading
from collections.abc import Generator

import gradio as gr
import matplotlib.pyplot as plt

from atmos20 import AtmosphereModel, ModelConfig
from atmos20.render import VARIABLES, plot_state, plot_vertical_section, status_markdown


_LOCK = threading.RLock()
_MODEL = AtmosphereModel()


def _parse_level(value: str | int | float) -> int:
    if isinstance(value, str):
        return int(value.split()[0])
    return int(value)


def _resolution(value: str) -> float:
    if value.startswith("1°"):
        return 1.0
    return 2.5 if value.startswith("2.5") else 5.0


def _time_step_seconds(resolution_deg: float) -> float:
    if resolution_deg <= 1.0:
        return 120.0
    return 300.0 if resolution_deg <= 2.5 else 600.0


def reset_model(
    resolution: str,
    tibet_scale: float,
    land_heating: float,
    currents: float,
    level: str,
    variable: str,
    vector_stride: int,
    section_lon: float,
):
    global _MODEL
    with _LOCK:
        resolution_deg = _resolution(resolution)
        config = ModelConfig(
            dlon_deg=resolution_deg,
            dlat_deg=resolution_deg,
            dt_seconds=_time_step_seconds(resolution_deg),
            tibet_height_scale=float(tibet_scale),
            land_heating_scale=float(land_heating),
            ocean_current_scale=float(currents),
        )
        _MODEL = AtmosphereModel(config)
        fig = plot_state(_MODEL, _parse_level(level), variable, vector_stride)
        section = plot_vertical_section(_MODEL, section_lon)
        return fig, section, status_markdown(_MODEL)


def step_model(hours: float, level: str, variable: str, vector_stride: int, section_lon: float):
    with _LOCK:
        _MODEL.advance_hours(float(hours))
        fig = plot_state(_MODEL, _parse_level(level), variable, vector_stride)
        section = plot_vertical_section(_MODEL, section_lon)
        return fig, section, status_markdown(_MODEL)


def redraw(level: str, variable: str, vector_stride: int, section_lon: float):
    with _LOCK:
        fig = plot_state(_MODEL, _parse_level(level), variable, vector_stride)
        section = plot_vertical_section(_MODEL, section_lon)
        return fig, section, status_markdown(_MODEL)


def run_stream(
    hours_per_frame: float,
    frames: int,
    level: str,
    variable: str,
    vector_stride: int,
    section_lon: float,
) -> Generator[tuple[object, object, str], None, None]:
    for _ in range(int(frames)):
        with _LOCK:
            _MODEL.advance_hours(float(hours_per_frame))
            fig = plot_state(_MODEL, _parse_level(level), variable, vector_stride)
            section = plot_vertical_section(_MODEL, section_lon)
            text = status_markdown(_MODEL)
        yield fig, section, text
        plt.close(fig)
        plt.close(section)


def build_app() -> gr.Blocks:
    pressure_choices = [f"{int(p)} hPa" for p in _MODEL.pressure_hpa]
    initial_figure = plot_state(_MODEL, 850, VARIABLES[0], 3)
    initial_section = plot_vertical_section(_MODEL, 90.0)

    with gr.Blocks(title="Atmos20") as demo:
        gr.Markdown(
            "# Atmos20：20 层理想化夏季大气\n"
            "1000–50 hPa，每 50 hPa 一层。地形来自 ETOPO 2022；默认 2.5°，可切换到较慢的 1°最高分辨率。"
            "地形会屏蔽地下层、阻挡低层风并产生迎风抬升。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                resolution = gr.Radio(
                    ["1° maximum", "2.5° detailed", "5° realtime"],
                    value="2.5° detailed",
                    label="水平分辨率",
                )
                tibet_scale = gr.Slider(0.0, 1.4, value=1.0, step=0.05, label="青藏高原高度倍率")
                land_heating = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="大陆夏季加热倍率")
                currents = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="洋流海温异常倍率")
                level = gr.Dropdown(pressure_choices, value="850 hPa", label="显示等压面")
                variable = gr.Dropdown(VARIABLES, value=VARIABLES[0], label="显示变量")
                vector_stride = gr.Slider(1, 6, value=3, step=1, label="风矢量稀疏度")
                section_lon = gr.Slider(0, 355, value=90, step=5, label="垂直剖面经度")
                hours = gr.Slider(0.5, 24.0, value=3.0, step=0.5, label="单步推进小时")
                frames = gr.Slider(1, 60, value=20, step=1, label="连续运行帧数")

                with gr.Row():
                    reset_btn = gr.Button("重建模型", variant="secondary")
                    step_btn = gr.Button("推进一步", variant="primary")
                with gr.Row():
                    run_btn = gr.Button("连续实时运行", variant="primary")
                    stop_btn = gr.Button("停止", variant="stop")

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("水平流场"):
                        plot = gr.Plot(value=initial_figure, label="流场")
                    with gr.Tab("20 层垂直剖面"):
                        section_plot = gr.Plot(value=initial_section, label="垂直剖面")
                status = gr.Markdown(status_markdown(_MODEL))

        reset_btn.click(
            reset_model,
            inputs=[resolution, tibet_scale, land_heating, currents, level, variable, vector_stride, section_lon],
            outputs=[plot, section_plot, status],
        )
        step_btn.click(
            step_model,
            inputs=[hours, level, variable, vector_stride, section_lon],
            outputs=[plot, section_plot, status],
        )
        level.change(redraw, inputs=[level, variable, vector_stride, section_lon], outputs=[plot, section_plot, status])
        variable.change(redraw, inputs=[level, variable, vector_stride, section_lon], outputs=[plot, section_plot, status])
        vector_stride.release(redraw, inputs=[level, variable, vector_stride, section_lon], outputs=[plot, section_plot, status])
        section_lon.release(redraw, inputs=[level, variable, vector_stride, section_lon], outputs=[plot, section_plot, status])

        run_event = run_btn.click(
            run_stream,
            inputs=[hours, frames, level, variable, vector_stride, section_lon],
            outputs=[plot, section_plot, status],
            stream_every=0.15,
        )
        stop_btn.click(fn=None, cancels=[run_event])

    return demo


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch()
