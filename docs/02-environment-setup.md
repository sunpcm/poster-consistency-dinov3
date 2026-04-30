# 02 — 环境搭建 SOP

> 从零(或半成品状态)把开发环境搭起来。**所有命令在 5090 服务器上执行**。
> 本项目当前状态见 [`AGENTS.md`](../AGENTS.md) 末尾的"项目状态快照"。

---

## 0. 前置检查(每次开工都跑一遍)

```bash
cd /opt/workspace/yongli/poster-consistency-dinov3

# A) 磁盘剩余 — < 20GB 立即停手(AGENTS.md R3)
df -h /

# B) GPU 可见
nvidia-smi -L
# 预期:GPU 0: NVIDIA GeForce RTX 5090 D ...

# C) tmux 守护(任何 >2min 的任务必须在 tmux 内)
tmux new -s dino   # 或 tmux attach -t dino
```

---

## 1. 创建 / 激活 venv

**已完成,跳过此步**。仅在新机器或 venv 损坏时执行:

```bash
uv venv .venv --python 3.10
```

激活:
```bash
source .venv/bin/activate
which python   # 必须指向 .venv/bin/python
```

---

## 2. 注入网络与缓存环境变量

每次新 shell **��须** 加载 `.env`(项目根),否则 HF/uv 会走默认源,大概率超时。

```bash
set -a && source .env && set +a

# 验证
echo $HF_ENDPOINT       # https://hf-mirror.com
echo $HF_HOME           # /opt/workspace/yongli/poster-consistency-dinov3/.hf_cache
echo $UV_CACHE_DIR      # /opt/workspace/yongli/poster-consistency-dinov3/.uv_cache
echo $UV_HTTP_TIMEOUT   # 300
```

`.env` 模板见 [`.env.example`](../.env.example)。

---

## 3. 安装 PyTorch(已完成)

**已完成**:`torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / torchaudio 2.6.0+cu124`

仅供参考,重装命令(双源策略,清华为主、PyTorch 官方为辅):

```bash
uv pip install torch torchvision torchaudio \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

> ⚠️ 如果遇到 `numpy timeout` 错误:`export UV_HTTP_TIMEOUT=300` 后重试。
> ⚠️ **不要**只用 `--index-url https://download.pytorch.org/whl/cu124`——该源只有 PyTorch 家族,numpy 等会失败。

---

## 4. 安装 HF 生态 + LoRA + 视觉(待执行)

⚠️ **执行前先告知用户**(AGENTS.md R2)。

```bash
uv pip install transformers accelerate peft huggingface_hub bitsandbytes opencv-python
```

预期总下载量 ~500MB,耗时约 1 分钟(走清华源)。

### 验证安装

```bash
python - <<'PY'
import torch, transformers, peft, accelerate, bitsandbytes, cv2
from huggingface_hub import __version__ as hf_v
print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"peft         {peft.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"bitsandbytes {bitsandbytes.__version__}")
print(f"hf_hub       {hf_v}")
print(f"opencv       {cv2.__version__}")
print(f"GPU          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
PY
```

预期 GPU 行:`NVIDIA GeForce RTX 5090 D`。

---

## 5. (可选)生成 pyproject.toml 与锁定文件

把项目升级为 uv "project" 形态,便于跨机复现:

```bash
# 在项目根
uv init --no-readme --no-pin-python   # 不要覆盖现有 README

# 然后手动编辑 pyproject.toml,加入 PyTorch 源锁定(见模板)
# 最后:
uv lock
```

**pyproject.toml** 推荐内容见 [`pyproject.toml`](../pyproject.toml)(若已生成)。关键片段:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

> ⚠️ **不要**把 `uv pip sync` 当成"安装命令"。它会**卸载** lock 之外的所有包。详情见 `image2xml/AGENTS.md` 的 2026-04-28 事故。

---

## 6. 常用诊断片段

### 6.1 看哪个 Python / pip 在工作

```bash
which python pip uv
python -c "import sys; print(sys.executable)"
```

### 6.2 看 torch 是否真能用 GPU

```bash
python -c "import torch; x=torch.randn(2,2).cuda(); print(x.device, x@x.T)"
```

### 6.3 列出本地���装的关键包及版本

```bash
uv pip list | grep -iE "^(torch|transformers|peft|accelerate|huggingface|bitsandbytes|opencv|pillow|numpy)"
```

### 6.4 清缓存(磁盘吃紧时)

```bash
# 交互式清 HF 历史版本
huggingface-cli delete-cache

# 清 uv 旧 wheel
uv cache clean
```

---

## 7. 故障字典

| 现象 | 原因 | 解法 |
|---|---|---|
| `Failed to download numpy ... network timeout` | uv 默认 30s 熔断 | `export UV_HTTP_TIMEOUT=300` |
| `huggingface-cli: command not found` | 没装 huggingface_hub 或没激活 venv | 走 §4 |
| `torch.cuda.is_available() == False` | 装到错误 Python / 装了 CPU 版 | `which python`,确认在 .venv;重装走 §3 双源 |
| `HF connection reset / 443 timeout` | 没设 HF_ENDPOINT | 走 §2,`echo $HF_ENDPOINT` |
| `OSError: ... model_index.json` | 模型路径错或下载未完成 | 见 `03-model-download.md` |
