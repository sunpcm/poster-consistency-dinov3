# 06 — 已知坑与避雷清单(Pitfalls)

> 来自 [`chat.md`](../chat.md) 的提炼 + 通用工程经验。**Phase 1 开工前先扫一遍**。
> 严重程度:🚨 P0(会直接把项目带沟里) / ⚠️ P1(会让结果不对但不会爆) / 💡 P2(优化提示)

---

## 🚨 P0-1:海报 Resize 形变破坏几何结构

**坑**:`AutoImageProcessor` 默认把图像直接缩放到方形(如 518×518)。海报通常是 16:9 或 9:16,这种暴力缩放会**彻底改变物体的长宽比**,DINOv3 预训练时从未见过这种"被压扁"的物体,特征质量直接崩。

**症状**:同一海报的不同角度照片,cosine 相似度低得离谱(< 0.6)。

**对策**:**aspect-ratio preserving padding**。

```python
import cv2
import numpy as np

def resize_with_padding(img: np.ndarray, target: int = 518, pad_value: int = 0) -> np.ndarray:
    """
    保持长宽比缩放,长边到 target,短边四周对称 padding。
    img: HxWx3 uint8 (BGR or RGB,自洽即可)
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = target - new_h
    pad_w = target - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[pad_value] * 3
    )
```

**集成方式**:在送入 `processor` 之前先 padding,然后 `processor(do_resize=False, do_center_crop=False, ...)`。

---

## 🚨 P0-2:`uv pip sync` 误伤共用 Python

**坑**:`uv pip sync requirements.lock` 不带 `--python` 时,会找系统第一个 Python 当目标。本机有 conda 共用安装,后果灾难性(参考 `image2xml/AGENTS.md` 的 2026-04-28 事故)。

**对策**:
- **本项目根本不要用 `uv pip sync`**,只用 `uv pip install`
- 真要用 sync,必须 `--python .venv/bin/python` 或先 `source .venv/bin/activate`

---

## 🚨 P0-3:HF 符号链接降级为物理复制

**坑**:`huggingface-cli download --local-dir <X>` 如果 `$HF_HOME` 与 `<X>` 不在同一物理分区,符号链接会变物理复制,空间翻倍。

**症状**:`du -sh models/` 与 `du -sh $HF_HOME/hub/` 加起来等于实际下载量的 2 倍。

**对策**:HF_HOME 和 models/ 都设在项目目录下,同盘。`.env` 已经这样配置。

详情见 `03-model-download.md` §4。

---

## 🚨 P0-4:磁盘爆满 → OS Kill

**坑**:根分区已用 95%(剩 105GB),HF 缓存 + LoRA checkpoint + 训练中间产物可以瞬间吃光。一旦满了,训练进程被 OOM Killer 干掉,可能损坏 checkpoint 文件。

**对策**:
1. **每次开工** `df -h /`,< 20GB 立即停手
2. 训练脚本**强制** `TrainingArguments(save_total_limit=3)`
3. 定期 `huggingface-cli delete-cache` + `uv cache clean`
4. 全部缓存重定向到项目目录(`.env` 已配置)

---

## ⚠️ P1-1:GAP 抹平局部异常(度量盲区)

**坑**:全局平均池化对**整体语义**敏感,对**局部缺陷**(漏印、缺角、墨点)几乎不敏感。Phase 1 基线会出现"明显有缺陷的 Logo cosine 还在 0.92"的反直觉结果。

**这是已知的、刻意保留的基线缺陷**,不是 bug。Phase 2 切到 Patch-EMD 解决。

详见 `01-business-context.md` §3.1。

---

## ⚠️ P1-2:Triplet Loss 模式崩溃

**坑**:Hard Negative 挖掘策略一旦没调好(尤其是"轻微缺陷 vs 严重形变"边界附近),余弦相似度 0.9 附近会剧烈振荡,模型不收敛。

**对策**:Phase 3 微调**优先 InfoNCE / SupCon**,Triplet 仅作消融对照。

---

## ⚠️ P1-3:合成形变增强的"业务毒性"

**坑**:用 TPS / 仿射变换在算法层模拟"褶皱",喂给模型作为正样本。模型学到的是**算法噪声特征**,不是真实布料物理。

**对策**:
- 优先采集**真实穿着 / 海报实拍**作为训练数据
- 合成增强仅用于早期 warmup 或扩充样本量
- 一定要做**真实数据 vs 合成数据**的消融实验

---

## ⚠️ P1-4:`uv pip install` 默认 30s 超时

**坑**:PyTorch wheel 几百 MB,国内网络下首次拉取超过 30s 是常态,uv 直接熔断。

**症状**:
```
Failed to download `numpy==2.2.6`
Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).
```

**对策**:`.env` 里已经设 `UV_HTTP_TIMEOUT=300`。每次开工 `set -a && source .env && set +a`。

---

## ⚠️ P1-5:RTX 5090 D 与 PyTorch 兼容性

**坑**:5090 D 是新卡(Blackwell 架构,sm_120),PyTorch 2.6.0+cu124 的 CUDA kernel 编译时**未必**显式包含 sm_120。如果遇到 `no kernel image is available for execution on the device`,需要升 PyTorch 到 nightly 或 cu126。

**当前状态**:已用 cu124 装好。**首次跑** `python -c "import torch; torch.randn(2,2).cuda() @ torch.randn(2,2).cuda()"` **必须验证**。如果报错,升级方案:
```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

---

## 💡 P2-1:`torch.compile` 双刃剑

**好处**:在 5090 上能让 DINOv3 推理吞吐翻倍。
**坑**:
- 第一次调用编译要几秒~几十秒(误以为卡死)
- 输入形状变化会触发重新编译
- 调试时报错堆栈不友好

**对策**:Phase 1 不用,Phase 2 优化时再开,且**必须 warmup 2-3 次**再计时。

---

## 💡 P2-2:`pooler_output` vs `last_hidden_state`

**坑**:DINOv3 自监督训练**没有专门的 pooler 头**。直接用 `outputs.pooler_output` 可能拿到的是恒等映射或简单的 CLS token,语义不如自定义池化。

**对策**:统一用 `outputs.last_hidden_state` + 自己池化(Phase 1 GAP / Phase 2 Patch-EMD)。

---

## 💡 P2-3:`bitsandbytes` 4-bit 加载

**坑**:7B + 4bit 在显存上节约,但 4bit 量化会**轻微降低特征质量**(对相似度比对场景影响约 1-3%)。

**对策**:7B 推理跑 fp16/bf16(5090 24GB+ 完全够);4bit 仅用于 LoRA 微调时压缩底座。

---

## 💡 P2-4:tmux 会话丢失

**坑**:`tmux new -s dino` 后忘记保存的输出,断连后 `tmux ls` 找不到,误以为任务挂了。

**对策**:
- 命名规范化:`dino-download` / `dino-train` / `dino-eval`
- 重要任务输出 tee 到日志:`python train.py 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M).log`
- `tmux ls` 永远先看一眼

---

## 💡 P2-5:Hugging Face token 泄漏

**坑**:`huggingface-cli login` 会把 token 写进 `~/.cache/huggingface/token`,公司共用机器其他用户能读。

**对策**:
- DINOv3 是公开模型,**不需要 login**
- 真要 login,用 `--add-to-git-credential` 时仔细考虑作用域
- 永远不要把 token 写到代码或 commit 里
