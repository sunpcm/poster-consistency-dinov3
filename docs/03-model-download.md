# 03 — 模型下载 SOP

> DINOv3 权重的下载、目录布局、避坑。先读 [`AGENTS.md`](../AGENTS.md) R3(磁盘红线)。

---

## 0. 下载前检查清单

```bash
df -h /                       # 剩余 > 25GB(下 Large+7B 都够);< 20GB 停手
echo $HF_ENDPOINT             # 必须是 https://hf-mirror.com
echo $HF_HOME                 # 必须在项目目录下,不能是 ~/.cache/huggingface
ls models/ 2>/dev/null        # 看是否已存在
```

如果 `HF_HOME` 没设,看 `02-environment-setup.md` §2。

---

## 1. ⚠️ 关键:模型仓库 ID 需要核对

`chat.md` 中 Gemini 写的是:
```
facebook/dinov3-large
facebook/dinov3-7b
```

**这是 AI 幻觉的可能命中点**。Meta DINOv3 在 HF 上的真实仓库命名约定通常是:
```
facebook/dinov3-vits16-pretrain-lvd1689m       # Small
facebook/dinov3-vitb16-pretrain-lvd1689m       # Base
facebook/dinov3-vitl16-pretrain-lvd1689m       # Large (300M)
facebook/dinov3-vith16plus-pretrain-lvd1689m   # Huge+
facebook/dinov3-vit7b16-pretrain-lvd1689m      # 7B 旗舰
```

**首次下载前必做**:

```bash
# A) 直接搜
huggingface-cli search facebook/dinov3 2>/dev/null | head

# B) 或浏览器访问
# https://huggingface.co/facebook?search_models=dinov3

# C) 或查官方 repo README
# https://github.com/facebookresearch/dinov3
```

**确认到正确 ID 后,把它写到 `models/MODEL_IDS.md`**(下面会建)以备后续参考。

---

## 2. 下载 DINOv3-Large(300M, ~1.2GB)

```bash
# 进 tmux(AGENTS.md R7)
tmux new -s dino-download

# 加载环境
cd /opt/workspace/yongli/poster-consistency-dinov3
source .venv/bin/activate
set -a && source .env && set +a

# 创建目标目录
mkdir -p ./models/dinov3-large

# 下载(把 <CONFIRMED_ID> 替换为 §1 确认的真实 ID)
huggingface-cli download <CONFIRMED_ID> \
  --local-dir ./models/dinov3-large \
  --exclude "*.msgpack" "*.h5" "*.ot" \
  --resume-download
```

参数说明:
- `--local-dir`:把模型放到我们的项目目录(磁盘可控)
- `--exclude`:跳过 Flax/TF/旧格式权重,只留 PyTorch safetensors,可省 30-50% 空间
- `--resume-download`:断点续传(网络抖动友好)

### 验证下载完整

```bash
ls -lh models/dinov3-large/
# 应包含:config.json / preprocessor_config.json / model.safetensors(或 pytorch_model.bin)

# 总大小
du -sh models/dinov3-large/
```

---

## 3. 下载 DINOv3-7B(可选,Phase 3 才用)

> **不要现在下**。当前磁盘只剩 ~105GB,7B 占 ~15GB,留给 Phase 3 再下。

预留命令:
```bash
mkdir -p ./models/dinov3-7b
huggingface-cli download <CONFIRMED_7B_ID> \
  --local-dir ./models/dinov3-7b \
  --exclude "*.msgpack" "*.h5" "*.ot" "*.bin" \
  --resume-download
# 注意 7B 推荐只留 safetensors(--exclude "*.bin")
```

---

## 4. 🚨 符号链接陷阱(必读)

`huggingface-cli download --local-dir <X>` 的真实行为:

1. 文件先下载到 `$HF_HOME/hub/models--<org>--<name>/blobs/`
2. 在 `<X>/` 下创建**符号链接**指向那些 blob

**陷阱**:如果 `$HF_HOME` 和 `<X>` **不在同一个物理磁盘分区**,符号链接会自动**降级为物理复制**——你以为节省了空间,实际上 1 个 15GB 的 7B 模型会吃掉 30GB!

**本项目对策**:`HF_HOME` 设在 `/opt/workspace/yongli/poster-consistency-dinov3/.hf_cache`,`models/` 也在同一项目目录下,**同盘**(都在根分区),符号链接安全。

**验证方式**:
```bash
ls -l models/dinov3-large/model.safetensors
# 期望输出:lrwxrwxrwx ... model.safetensors -> /opt/workspace/yongli/.../.hf_cache/...
# 如果是 -rw-r--r--(普通文件)且大小 = 真实权重大小,说明符号链接降级了!
```

---

## 5. 离线加载验证

```bash
python - <<'PY'
from transformers import AutoImageProcessor, AutoModel
path = "./models/dinov3-large"
proc = AutoImageProcessor.from_pretrained(path)
model = AutoModel.from_pretrained(path)
print("✅ Loaded.")
print(f"   processor: {type(proc).__name__}")
print(f"   model:     {type(model).__name__}")
print(f"   hidden:    {model.config.hidden_size}")
PY
```

预期输出包含 `hidden: 1024`(Large)或 `hidden: 768`(Base)。

---

## 6. 磁盘预算速查

| 模型 | 占用(safetensors,fp16/bf16) | 何时下 |
|---|---|---|
| DINOv3-Small | ~80MB | 仅消融实验 |
| DINOv3-Base | ~340MB | 备用 |
| **DINOv3-Large** | **~1.2GB** | **Phase 1 主用** |
| DINOv3-Huge+ | ~2.5GB | 可选 |
| DINOv3-7B | ~14GB | Phase 3 才下 |

加上 HF blob 缓存(同盘符号链接情况下不重复占用),Phase 1 总开销 < 2GB。

---

## 7. 删除模型 / 释放空间

```bash
# 安全方式:先删 local-dir,再清缓存
rm -rf models/dinov3-large
huggingface-cli delete-cache   # 交互式选择仓库
```

**不要** `rm -rf $HF_HOME` 直接清,会破坏 HF 的索引,后续下载需要重建。
