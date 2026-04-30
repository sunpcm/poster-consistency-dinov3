# AGENTS.md — poster-consistency-dinov3

> 工作在 **公司共用 5090 服务器** 上的硬规与协作约束。任何 AI agent 接手本项目前**必读**。
> 如果你是人类,也请通读一遍。

---

## 🎯 项目目的(一句话)

用 **DINOv3 自监督视觉特征** 做 **服装 / 海报 Logo 在柔性形变下的一致性检测**——识别错版、漏印、墨点、缺笔等异常,同时容忍褶皱、光影、拉伸等物理变形。

详细业务背景见 [`docs/01-business-context.md`](./docs/01-business-context.md)。

---

## 🖥️ 运行环境(必知)

| 项 | 值 |
|---|---|
| 机器 | 公司 5090 服务器(`/opt/workspace/yongli/poster-consistency-dinov3`) |
| GPU | **NVIDIA RTX 5090 D**(单卡) |
| 网络 | **纯净环境,无 mihomo / 无透明代理**——和 `image2xml` 不同,不要套用那边的网络红线 |
| Python | 3.10.18(由 uv 管理) |
| 包管理 | **uv 0.7.17**,**不使用 conda** |
| 磁盘 | 根目录 1.8T,已用 95%,**仅剩 ~105GB**——关键约束,见 R3 |
| 开发方式 | Mac(M1 Pro) + 1Password SSH Agent + VS Code Remote-SSH + tmux 守护 |

---

## 🚨 硬规则(违反 = 任务失败)

### R1. 环境隔离:只写 `.venv/`
本机是公司共用服务器,禁止污染全局:

- 所有 Python 依赖**只能装在** `/opt/workspace/yongli/poster-consistency-dinov3/.venv/`
- **禁止** `pip install`(系统/全局 pip)、`conda install`、`apt install`(除非用户明确许可)
- **禁止** 在 `~/` 下堆缓存——所有缓存必须重定向到项目目录(见 R3)

### R2. 任何写环境/系统的操作必须先获得用户许可
触发条件(任一即需先问):
- `uv pip install` / `uv pip sync` / `uv add` / `uv sync`
- `apt` / `npm install -g` / 任何系统级包管理
- 改 `PATH` / shell 配置 / systemd / `/etc/`
- 写入 `/usr/`、`/opt/` 之外(项目目录除外)
- **下载 > 1GB 的模型权重** → 必须先确认磁盘(见 R3)
- 运行 `huggingface-cli login` 或涉及 token 的命令

### R3. 磁盘红线:<20GB 时拉警报
当前 `/` 仅剩 ~105GB,被公司其他业务共用。硬性约束:

- 任何一次操作前,先 `df -h /` 确认剩余
- **剩余 < 20GB** → **立即停手,告知用户**,不要继续下载/训练
- HF 缓存 + uv 缓存 + 模型权重必须全部落在 `/opt/workspace/yongli/poster-consistency-dinov3/` 下,**禁止**写入 `~/.cache/`
- LoRA 训练必须强制 `save_total_limit=3`,防止 checkpoint 堆爆

**静态存储预算**(见 `docs/03-model-download.md` 展开):
- 环境 5-8GB + DINOv3-Large 1.2GB + DINOv3-7B 14-15GB + 数据/权重 2-5GB ≈ **25-30GB**
- 留 70GB 给训练缓存、checkpoint、数据增强

### R4. uv 调用模板(项目内唯一允许的形式)

```bash
# 创建 venv(已完成,不要重跑)
uv venv .venv --python 3.10

# 装依赖 — 必须显式指定目标,或先 source 激活
source .venv/bin/activate
uv pip install <pkg>

# 或用 VIRTUAL_ENV 显式锚定
VIRTUAL_ENV=$(pwd)/.venv uv pip install <pkg>
```

**禁止形式**:
- ❌ `uv pip sync requirements.lock`(无 `--python`,会把系统第一个 Python 当目标)
- ❌ 未激活 venv 就跑 `uv pip install`
- ❌ `pip install`(不走 uv)

> **注**:关于 `uv pip sync` 的灾难性风险,详见 `../image2xml/AGENTS.md` 的 2026-04-28 事故记录。不要在本项目重演。

### R5. 网络与镜像:这台机器的特殊性
这台 5090 服务器**没有** mihomo 透明代理,但**国内访问 HF/PyPI 会超时**。因此:

- ✅ **允许**并**推荐** `export HF_ENDPOINT=https://hf-mirror.com`(会话级)
- ✅ **允许** `export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`
- ✅ **允许** `export UV_HTTP_TIMEOUT=300`(PyTorch wheel 大,默认 30s 会熔断)
- ❌ **禁止**在 Docker/systemd 层注入 `HTTP_PROXY`/`HTTPS_PROXY`(这台机器没代理,写了只会坏事)
- ❌ **禁止**改 `/etc/resolv.conf` 或系统级镜像配置(污染其他用户)

所有网络环境变量**只在当前 Session 或项目 `.env` 生效**,不写 `~/.bashrc`(避免影响其他项目)。

### R6. 凭据安全
- Hugging Face token、OpenAI key 等**绝不**明文落盘到服务器
- SSH key 由 Mac 端 1Password SSH Agent 透传,服务器端**不**存 private key
- `.ssh_key*`、`.env`、`*.token` 已在 `.gitignore`,提交前必 `git status` 确认

### R7. 长任务必须 tmux
Mac 合盖 / 切 WiFi 会断 SSH,进程被 kill。所以:

- 任何耗时 > 2 分钟的命令(模型下载、训练、批量推理)→ **必须** `tmux new -s <name>`
- 推荐命名:`tmux new -s dino-download` / `tmux new -s dino-train`

---

## 📜 协作风格约定

从 `chat.md` 对话和用户偏好提炼:

- **中文交流**,代码注释和术语保留英文
- **简洁直接**,不要客套(不要 "Great question!" / "让我来..." / "I'm on it")
- **任何写操作先报告意图再执行**,特别是环境/系统级 → R2
- **客观指出盲区**,不盲目认同用户或其他 AI 的方案
  - 例:`chat.md` 里 Gemini 推荐 GAP,本项目明确把 GAP 当**基线**而非**终局**,Phase 2 切换到 Patch-EMD
- **出错立即停手**,承认错误,给诊断和方案让用户决策
- **提供选项时默认推荐项要明确标注**("推荐 A,因为 X")

---

## 🗺️ AI 接手项目必读顺序

1. **本文件(AGENTS.md)** — 硬规则与环境
2. **[`README.md`](./README.md)** — 项目目的、快速上手
3. **[`docs/01-business-context.md`](./docs/01-business-context.md)** — 业务目标、算法选型与决策
4. **[`docs/05-roadmap.md`](./docs/05-roadmap.md)** — 当前处于哪个阶段
5. **[`docs/06-pitfalls.md`](./docs/06-pitfalls.md)** — 已知坑,动手前扫一遍
6. **[`chat.md`](./chat.md)** — 原始讨论,保留以备追溯(**不是**真理来源)

执行类文档按需查:
- 环境搭建 → `docs/02-environment-setup.md`
- 模型下载 → `docs/03-model-download.md`
- 推理验证 → `docs/04-verification-script.md`

---

## 📜 事故记录

_(目前无。若发生,按 `image2xml/AGENTS.md` 的格式追加:起因 / 症状 / 影响范围 / 恢复方案 / 关键坑 / 根因教训)_

---

## 📌 当前项目状态快照(2026-04-29)

- ✅ uv venv 已建(Python 3.10.18)
- ✅ PyTorch 2.6.0+cu124 + torchvision 0.21.0+cu124 + torchaudio 已装
- ✅ Pillow 12.1.1 已装
- ⏳ **未装**:transformers / accelerate / peft / huggingface_hub / bitsandbytes / opencv-python
- ⏳ **未下载**:DINOv3-Large、DINOv3-7B
- ⏳ **未编写**:`verify_dinov3.py`
- ⏳ **未生成**:`pyproject.toml`、`uv.lock`

下一步见 `docs/05-roadmap.md` 的 Phase 1。
