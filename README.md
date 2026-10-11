# AI 原生视频张量编辑引擎

视频解码为 PyTorch 裸张量，全程在张量层面做切帧、改像素、拼接、调色、转场，
再编码回标准 MP4。DeepSeek API 通过 Function Calling 指挥张量操作。
不靠截屏、不靠点击 GUI、不依赖任何商业剪辑软件。

## 项目结构

```
tensor-video-editor/
├── tensor_editor/
│   ├── __init__.py      # 公开 API
│   ├── decode.py        # 模块1：FFmpeg rawvideo 管道 -> [T,H,W,C] uint8 张量
│   ├── edit_engine.py   # 模块2：TensorEditor（切帧/改像素/拼接/调色/转场/字幕/缩放）
│   ├── encode.py        # 模块3：张量 -> FFmpeg 管道 -> H.264 MP4
│   ├── ai_agent.py      # 模块4：AI 指挥官（Function Calling，多供应商可切换）
│   ├── providers.py     # 供应商注册表：DeepSeek/MiniMax/通义/Kimi/豆包
│   ├── template.py      # 模块5：JSON/YAML 模板（继承、覆盖、批量套用）
│   └── batch.py         # 模块6：多线程批量处理
├── config.py            # 模块7：环境变量 + 命令行参数
├── main.py              # 模块7：入口，一键批量
├── example.py           # 最小可运行示例
├── template.json        # 示例模板
├── requirements.txt
└── tests/               # pytest 单元测试
```

## 安装步骤

### 1. 安装 FFmpeg（必须，pip 装不了）

```bash
# Ubuntu/Debian
sudo apt install ffmpeg
# macOS
brew install ffmpeg
# 验证
ffmpeg -version
```

### 2. 创建虚拟环境并安装依赖

```bash
cd tensor-video-editor
python3 -m venv .venv
source .venv/bin/activate

# 先装 CPU 版 torch（体积小很多；需要 GPU 版就去 pytorch.org 按指引装）
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 国内网络慢可用阿里镜像：
# pip install torch --index-url https://mirrors.aliyun.com/pytorch-wheels/cpu

pip install -r requirements.txt
```

### 3. 配置 AI Key（只在用 AI 指挥时需要）

`--ai` 模式需要对应供应商的 API Key，Key 只从环境变量读取，不会写入任何代码或文件：

| `--provider` | 供应商 | 默认模型 | 环境变量 |
|---|---|---|---|
| `deepseek`（默认） | DeepSeek | deepseek-chat | `DEEPSEEK_API_KEY` |
| `minimax` | MiniMax | MiniMax-M2.7 | `MINIMAX_API_KEY` |
| `qwen` | 通义千问 | qwen-plus | `DASHSCOPE_API_KEY` |
| `kimi` | Kimi | kimi-k2-0905-preview | `MOONSHOT_API_KEY` |
| `doubao` | 豆包（火山方舟） | 无（须 `--model` 给接入点 ID） | `ARK_API_KEY` |

```bash
export DASHSCOPE_API_KEY=sk-xxx   # 示例：用通义千问
```

## 运行示例

### 最小示例（不需要 API Key）

```bash
python example.py
# 自动用 FFmpeg 生成 sample.mp4（testsrc，3 秒）
# 解码 -> 切10帧 -> 第5帧画红框 -> 拼回去 -> 输出 example_out.mp4
```

### 一键批量（模板驱动，不需要 API Key）

```bash
# 准备：把视频放进 ./videos，模板用自带的 template.json
python main.py --input ./videos --output ./output --template ./template.json

# 常用参数
python main.py --input ./videos --output ./output --template ./template.json \
    --workers 4 --width 1280 --height 720 --fps 30
```

### 启用 AI 指挥（需要对应供应商的 API Key）

```bash
# 通义千问
export DASHSCOPE_API_KEY=sk-xxx
python main.py --input ./videos --output ./output --template ./template.json \
    --ai --provider qwen

# MiniMax / Kimi / DeepSeek 同理，换 --provider 和对应的环境变量
# 豆包：model 必须填火山方舟控制台创建的推理接入点 ID
export ARK_API_KEY=xxx
python main.py --input ./videos --output ./output --template ./template.json \
    --ai --provider doubao --model ep-xxxxxxxx
```

`--ai` 会把模板里的 `ai_instruction`（如"去掉右上角水印，整体调暖色调"）
送给所选供应商，由它通过 Function Calling 指挥张量操作。
可用 `--model` / `--base-url` 覆盖供应商默认值。

### 跑测试

```bash
pytest tests/ -v
```

## 模板配置说明

`template.json` 示例：

```json
{
  "ai_instruction": "去掉右上角水印，整体调暖色调",
  "color": {"brightness": 1.1, "contrast": 1.05, "saturation": 1.15},
  "segments": [{"start": 0, "end": 100}, {"start": 150, "end": 300}],
  "intro": null,
  "outro": null,
  "transition": "fade",
  "transition_duration": 12
}
```

| 字段 | 说明 |
|---|---|
| `segments` | 保留的片段 `[start, end)`（帧号），按顺序拼接；`null` 保留全部 |
| `color` | 调色，`1.0` 为不变 |
| `transition` | `fadein` / `fadeout` / `fade`，`transition_duration` 为帧数 |
| `intro` / `outro` | 片头/片尾视频路径（自动按主视频规格解码后拼接） |
| `ai_instruction` | 自然语言指令，仅 `--ai` 模式送 DeepSeek |
| `extends` | 继承另一个模板文件，子模板字段覆盖父模板（深合并） |

也支持 YAML 模板（`template.yaml`），字段相同。

## AI 供应商接入说明

- 所有供应商都走 **OpenAI 兼容接口**（`POST /v1/chat/completions` +
  Function Calling），`ai_agent.py` 的 `VideoAIAgent` 通过 `provider` 参数切换，
  供应商注册表见 `tensor_editor/providers.py`。新增一家只需在注册表加一条，
  不用改 Agent 逻辑。
- 6 个工具：`cut_frames` / `modify_region` / `concatenate_videos` /
  `apply_color` / `apply_transition` / `export_video`，全部符合
  OpenAI Function Calling 规范。
- Agent 循环：发指令+张量元信息（T/H/W/C/dtype）-> 执行 tool_calls ->
  结果回传 -> 直到无 tool_calls，最多 10 轮。
- 每条 AI 指令都会打日志（`logging`）。
- 无 Key 时会抛 `AgentError` 明确提示缺哪个环境变量，不会静默失败。

已知坑（实测文献，非猜测）：
- **MiniMax** 的 Function Calling 对多参数工具有时会返回空 `arguments`
 （上游模型行为，与本引擎无关）。如果 AI 指令执行异常，先换
  `--provider qwen` 或 `deepseek` 试试。
- **豆包** 的 `model` 参数填的是火山方舟控制台里创建的**推理接入点 ID**
  （如 `ep-xxx`），不是模型名；不填会直接报错提示。
- MiniMax 国际站地址是 `https://api.minimax.io/v1`，如需切换用
  `--base-url` 覆盖默认值。

## Python API 速览

```python
from tensor_editor import decode_video_to_tensor, encode_tensor_to_video, TensorEditor

frames = decode_video_to_tensor("in.mp4", 1280, 720, fps=30)  # [T,H,W,C] uint8
ed = TensorEditor(frames)
out = (ed.cut_frames(0, 150)
         .apply_color(brightness=1.1, saturation=1.2)
         .apply_transition("fade", duration=12))
encode_tensor_to_video(out.frames, "out.mp4", fps=30)
```

## 常见问题

**Q: pip install torch 太慢/超时？**
A: 用 CPU 源或国内镜像（见"安装步骤"第 2 步），或加 `--default-timeout 120`。

**Q: `DecodeError: 找不到 ffmpeg`？**
A: FFmpeg 必须单独安装并在 PATH 中，`ffmpeg -version` 验证。

**Q: 大视频内存爆了？**
A: 张量常驻内存，`[T,H,W,C]` uint8 每帧约 `H*W*3` 字节。
   1080p 30fps 1 分钟约 11GB。建议先 `--width 1280 --height 720` 降规格，
   或按 `segments` 分段处理。

**Q: DeepSeek 返回了奇怪的 tool_calls？**
A: `ai_agent` 对参数做校验，非法参数会以 `{"ok": false, ...}` 回传给模型
   让它修正；连续失败看日志里的 "AI 指令" 行。

**Q: `add_text` 中文乱码？**
A: 默认字体可能缺中文字形。传 `font_size` 调整，或自行指定 PIL 可加载的
   中文字体文件（改 `edit_engine.py` 的 `ImageFont.truetype`）。

**Q: 批量时某条失败？**
A: 默认重试 2 次仍失败会记录错误并继续下一条，最后汇总打印，不会整批崩。

## 验收对照

| # | 验收项 | 对应 |
|---|---|---|
| 1 | `python example.py` 跑通，输出可播放 | example.py + ffprobe 自检 |
| 2 | 解码→张量→改→编码无报错 | tests/test_decode.py, test_encode.py |
| 3 | AI 返回 tool_calls 并执行（任一 `--provider`） | tests/test_agent.py（打桩）+ 真 Key 手动跑 |
| 4 | 批量 10 条全部输出成功 | tests/test_template_batch.py + main.py |
| 5 | 切片/拼接/调色结果正确 | tests/test_edit.py |
| 6 | README 清晰可跑通 | 本文件 |
