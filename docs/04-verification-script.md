# 04 — verify_dinov3.py 推理验证脚本说明

> 这个脚本的目标:**端到端跑通"图像 → DINOv3 → 特征向量"**,不做任何业务判断。
> 它的成功 = 环境、模型、CUDA 调用、特征维度都正常。

文件位置:[`scripts/verify_dinov3.py`](../scripts/verify_dinov3.py)(待生成,见 `05-roadmap.md` Phase 1)。

---

## 1. 脚本骨架(参考)

```python
"""
verify_dinov3.py — 端到端推理验证

目标:
  1) 确认 DINOv3-Large 能从本地路径离线加载
  2) 确认 5090 GPU 可用,模型能推上去
  3) 确认前向传播能产出预期维度的特征
  4) 输出单张推理延迟,作为后续优化基线

使用:
  python scripts/verify_dinov3.py
  python scripts/verify_dinov3.py --image data/test_samples/some.jpg
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_PATH = Path(__file__).parent.parent / "models" / "dinov3-large"


def load_image(path: str | None) -> Image.Image | torch.Tensor:
    if path:
        return Image.open(path).convert("RGB")
    # fallback:随机张量(仅验证管线,不验证语义)
    return torch.randint(0, 255, (3, 518, 518), dtype=torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--compile", action="store_true", help="enable torch.compile")
    args = ap.parse_args()

    print(f"[*] Loading DINOv3 from: {MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[*] Mounted to: {device.upper()}  ({torch.cuda.get_device_name(0) if device=='cuda' else '-'})")

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")
        print("[*] torch.compile enabled (warmup needed)")

    image = load_image(args.image)
    print(f"[*] Input: {'real ' + args.image if args.image else 'random tensor 3x518x518'}")

    # warmup(尤其 torch.compile 模式)
    with torch.no_grad():
        for _ in range(2):
            inputs = processor(images=image, return_tensors="pt").to(device)
            _ = model(**inputs)

    # 计时
    t0 = time.time()
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        feats = outputs.last_hidden_state         # [1, N+1, D]  (含 CLS token)
        global_feat = feats.mean(dim=1)           # GAP → [1, D]
        global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=1)
    t1 = time.time()

    print("=" * 50)
    print(f"✅ DINOv3 推理链路贯通!")
    print(f"   last_hidden_state shape: {feats.shape}")
    print(f"   global feature shape:    {global_feat.shape}")
    print(f"   single inference latency: {(t1 - t0) * 1000:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

---

## 2. 关键点解读

### 2.1 为什么尺寸是 518?

DINOv3 ViT-L 的 patch size = 16,默认输入 518×518 → 32×32 + 1 CLS = 1025 tokens。
518 = 14 × 37,是预训练时常用的尺寸,**改尺寸会偏离预训练分布**。

### 2.2 GAP 仅作基线
脚本中:
```python
global_feat = feats.mean(dim=1)   # 简单平均所有 patch token + CLS
```

这是 [`01-business-context.md`](./01-business-context.md) §3.1 讨论的 **方案 A**。
**它能跑通管线,但不应作为最终度量**。Phase 2 切到 Patch-EMD,届时不要 mean,直接保留 `[1, N, D]` patch grid。

### 2.3 为什么用 last_hidden_state 而不是 pooler_output?
DINOv3 自监督预训练**没有专门的 pooler 头**(不像 CLIP),`pooler_output` 可能是恒等或简单池化。直接取 `last_hidden_state` 自己池化更可控。

### 2.4 为什么有 warmup?
- `torch.compile` 第一次调用会触发图编译,几秒~几十秒
- CUDA kernel 第一次调用有 cuBLAS / cuDNN 初始化开销
- **不 warmup 测出来的延迟会偏高 5-10 倍,是噪声**

### 2.5 ⚠️ 默认 processor 的 Resize 陷阱
`AutoImageProcessor` 默认会把图像**直接缩放到 518×518**——对长方形海报这是**形变破坏**!

Phase 1 验证时无所谓,但 Phase 1.5 起在 `src/feature_extract.py` 中**必须**改成:
1. 保持长宽比把长边缩到 518
2. 短边四周对称 padding(黑色或反射) 到 518
3. 再走 processor 的 normalize 步骤

代码模板见 [`06-pitfalls.md`](./06-pitfalls.md) §2。

---

## 3. 运行方式

```bash
cd /opt/workspace/yongli/poster-consistency-dinov3
source .venv/bin/activate
set -a && source .env && set +a

# 随机张量 smoke test
python scripts/verify_dinov3.py

# 真实图像
python scripts/verify_dinov3.py --image data/test_samples/logo_sample.jpg

# 开 torch.compile(Phase 2 优化时再用)
python scripts/verify_dinov3.py --compile
```

---

## 4. 期望输出

```
[*] Loading DINOv3 from: /opt/workspace/yongli/poster-consistency-dinov3/models/dinov3-large
[*] Mounted to: CUDA  (NVIDIA GeForce RTX 5090 D)
[*] Input: random tensor 3x518x518
==================================================
✅ DINOv3 推理链路贯通!
   last_hidden_state shape: torch.Size([1, 1025, 1024])
   global feature shape:    torch.Size([1, 1024])
   single inference latency: 25-50 ms
==================================================
```

5090 上 Large 单张 ~30ms,如果显著超过 100ms,检查:
- `device == 'cuda'`?
- `model.eval()` 调用了吗?
- 是否漏了 warmup?

---

## 5. 常见错误

| 错误 | 原因 | 解 |
|---|---|---|
| `OSError: ... model_index.json` | 模型路径错或下载未完成 | 见 `03-model-download.md` |
| `ValueError: Unrecognized image processor in <path>` | preprocessor_config.json 缺失 | 重下载,加 `--resume-download` |
| `RuntimeError: CUDA out of memory` | 多进程占用 / 之前的 Python 没释放 | `nvidia-smi` 看占用,`pkill -f python` 或重开 tmux |
| `RuntimeError: shape mismatch` 在 mean 上 | 模型输出格式与预期不符(版本差异) | `print(outputs)` 看实际字段名 |
