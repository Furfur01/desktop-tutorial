# Atmos20 — 气象模拟

[English README](README.md)

Atmos20 是一个交互式的理想化大气环流与天气系统模拟器。它会积分一个紧凑的 20 层干大气模型，把连续演化的模式状态渲染成浏览器可以直接播放的动态素材，再显示到可旋转的全球球面上。

> Atmos20 用来展示物理机制、数值过程和可视化流程。它不提供天气预报、再分析资料或气候预测。

## 这玩意是干什么的

目前仓库里有两类实验：

- **带真实地形的全球环流**：Held–Suarez 风格的温度松弛和近地面阻力，ETOPO 地形下边界，海陆温差、地形耦合，以及一个简化的三圈经向环流闭合。
- **干斜压波生命周期**：理想化中纬度不稳定实验，并从模式输出中诊断气旋、冷锋和暖锋。

每次计算都会生成互相同步的动态 WebP，包括风速、带符号纬向风、温度、气压异常、垂直运动、位势高度、适用场景下的锋面，以及随时间变化的粒子流线。浏览器只接收渲染素材和 JSON 清单，不会拿到完整的数值状态。

## 快速运行

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate        # Windows：.venv\Scripts\activate
python -m pip install -e .[dev]

python scripts/prerender_windy.py \
  --scenario circulation \
  --output web/assets/prerender \
  --grid-degrees 5 \
  --spinup-hours 120 \
  --analysis-hours 120 \
  --frames 48 \
  --fps 12 \
  --particles 2400 \
  --level 900

python windy_app.py
```

随后打开 `windy_app.py` 输出的地址。

## 手动计算与预渲染结果

左侧面板现在可以切换两种结果来源：

- **预渲染结果**：从 `web/assets/prerenders/catalog.json` 读取已经完成的计算包。加载速度快，也不会占用本机算力。
- **手动计算**：把面板里的参数提交给本地后端。后端完成积分、图层渲染、动态编码和质量检查后，页面会自动载入新结果。

仓库自带原先的 10 天默认动态结果。云端或本地生成其他方案并发布目录后，它们会自动出现在预渲染列表里。

## 长时间积分

预设方案直接用“天”描述时长，避免到处填写几百或几千小时。渲染器内部仍会把天数换算为秒，并沿同一条连续时间轨迹积分。

当前预设包括：

| 方案 | 网格 | 模拟时长 | 用途 |
|---|---:|---:|---|
| `circulation-default-10d` | 5° | 10 天 | 当前默认结果 |
| `circulation-long-equinox-90d` | 5° | 90 天 | 60 天建立期 + 30 天动态分析窗 |
| `circulation-long-summer-60d` | 5° | 60 天 | 北半球夏季热赤道 |
| `circulation-long-winter-60d` | 5° | 60 天 | 北半球冬季热赤道 |
| `circulation-quality-20d` | 2.5° | 20 天 | 更细的地形和风带结构 |
| `baroclinic-north-10d` | 5° | 10 天 | 北半球干斜压波生命周期 |
| `baroclinic-south-10d` | 5° | 10 天 | 南半球干斜压波生命周期 |

需要增加参数组合时，编辑 `prerender/profiles.json`。不建议把所有滑块做成笛卡尔积全扫一遍，计算量和媒体体积会很快失控。

## 用 GitHub Actions 做云端预渲染

`.github/workflows/prerender.yml` 只支持手动触发，避免没事自己烧算力。进入仓库的 **Actions** 页面，运行 **build pre-render library**，填写逗号分隔的方案 ID，或者填写 `all`。

工作流会：

1. 检查方案 ID；
2. 把互相独立的方案拆成受限的矩阵任务并行运行；
3. 让编译型数值库使用虚拟机提供的 CPU 线程；
4. 给每个任务设置 330 分钟上限；
5. 检查帧数、时间轴、环流质量门禁和真实的逐帧变化；
6. 把每个方案上传成短期工作流产物；
7. 选择发布时，把成功的结果和更新后的目录一次性提交回当前分支。

动态素材可能让仓库迅速变大。建议只运行确实要看的方案，并在启动高成本任务前检查仓库当时的 GitHub Actions 用量与计费规则。

## 本地多核计算

同一条大气时间轨迹存在严格的先后依赖，不能把前后时刻随便扔给多个进程各算各的。互相独立的实验方案可以安全并行。

```bash
python scripts/prerender_profiles.py list

python scripts/prerender_profiles.py render-many \
  --select circulation-long-equinox-90d,baroclinic-north-10d \
  --jobs 2
```

`--jobs` 控制独立工作进程数量。它不应超过方案数量；内存紧张时还要继续调低。GitHub Actions 中采用的是同一种粗粒度并行方式，只是每个方案分配到独立的矩阵虚拟机。

单独计算一个方案：

```bash
python scripts/prerender_profiles.py render \
  --profile circulation-long-equinox-90d
```

复制或删除结果后重建浏览器目录：

```bash
python scripts/prerender_profiles.py catalog
```

## 动态过程显示

现在的视觉结果来自连续的模式时次，不会拿一张冻结风场反复糊弄：

- 标量图层和粒子流线共享同一条模式时间轴；
- 动态 WebP 会逐帧解码，并同步上传为 WebGL 纹理；
- 底部时间轴可以直接拖到任意模式帧；
- 长积分按模拟天数显示，不再只给一个没有意义的帧编号；
- 浏览器缺少所需的 WebGL/WebCodecs 能力时，会退回二维原生动画。

发布前可以运行：

```bash
python scripts/validate_prerender.py \
  web/assets/prerenders/<方案ID>/manifest.json
```

检查器会拒绝缺文件、帧数不一致、时间轴倒退、环流门禁失败，以及抽样帧看起来完全相同的“假动画”。

## 数值与物理范围

当前模式采用：

- 1000–50 hPa 的 20 个固定气压层，间隔 50 hPa；
- 经纬度网格，并根据地形屏蔽地下气压层；
- SSPRK3 时间推进；
- MC 限制器二阶 TVD 水平平流；
- 随分辨率缩放的扩散、散度阻尼、垂直混合和简化干物理强迫；
- ETOPO 2022 地形下边界，以及 Natural Earth 1:110m 国界作为显示背景。

全球环流实验用于观察环流机制和地形耦合。斜压波实验用于观察不稳定发展与锋面诊断。当前配置没有湿对流、云微物理、辐射、资料同化或业务模式边界条件。

## 仓库结构

```text
src/atmos20/                  数值模式与诊断
scripts/prerender_windy.py    模式积分和动态素材渲染器
scripts/prerender_profiles.py 预设方案、多进程方案并行、目录生成
scripts/validate_prerender.py 动态素材与逐帧变化检查
prerender/profiles.json       长积分及代表性参数方案
windy_app.py                  本地 HTTP 服务和按需计算 API
web/                          全球球面界面与随仓库提供的素材
.github/workflows/            测试和手动云端预渲染工作流
```

## 测试

```bash
pytest
python scripts/prerender_profiles.py --registry prerender/profiles.json matrix --select all
python scripts/validate_prerender.py web/assets/prerender/manifest.json
```

最后一条会检查仓库自带的默认动画。

## 参考与数据来源

- Held, I. M. 与 Suarez, M. J.（1994），大气环流模式动力框架对比试验。
- Jablonowski, C. 与 Williamson, D. L.（2006），大气模式动力框架斜压不稳定测试。
- NOAA National Centers for Environmental Information，ETOPO 2022。
- Natural Earth 1:110m 行政区边界。

## 许可证

MIT，见 [LICENSE](LICENSE)。
