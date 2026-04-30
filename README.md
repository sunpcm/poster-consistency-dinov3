# poster-consistency-dinov3

> **服装 / 海报 Logo 在柔性形变下的一致性异常检测**
> 基于 Meta DINOv3 自监督视觉特征 · RTX 5090 · Python 3.10 + uv

---

## 一句话说明

给定一张"标准 Logo"和一张"待检 Logo(可能已印在皱巴巴的衣服或海报上)",判断两者是否是同一个 Logo,并定位错版、漏印、墨点、缺笔等异常——同时容忍褶皱、光影、拉伸等物理变形。

## 为什么用 DINOv3

- **自监督**,不依赖 Logo 标注数据,直接迁移
- **Patch 级语义特征** 对旋转/遮挡/形变有天然鲁棒性
- 300M(Large) 参数量在 5090 上推理 < 50ms / 张,吞吐够用
- 7B 旗舰版 + LoRA 可作为高精度兜底

对 GAP 方案的盲区分析、替代路线(Patch-EMD / Deformable Attention)详见 [`docs/01-business-context.md`](./docs/01-business-context.md)。

---

## 快速开始

> ⚠️ 开工前**必读** [`AGENTS.md`](./AGENTS.md),特别是 R3(磁盘)和 R5(网络)。

### 0. 进入项目并激活环境

```bash
cd /opt/workspace/yongli/poster-consistency-dinov3
source .venv/bin/activate

# 注入本项目的网络 / 缓存配置(会话级,不污染 shell)
set -a && source .env && set +a
```

### 1. 补齐依赖(只做一次)

```bash
# tmux 守护,防 Mac 断连
tmux new -s dino-setup

uv pip install transformers accelerate peft huggingface_hub bitsandbytes opencv-python
```

### 2. 下载 DINOv3-Large(约 1.2GB)

```bash
mkdir -p ./models/dinov3-large
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m \
  --local-dir ./models/dinov3-large \
  --resume-download
```

> ⚠️ **模型仓库名称需要核对**——Gemini 原文用的是 `facebook/dinov3-large`,但 HF 实际仓库命名约定是 `facebook/dinov3-vitl16-pretrain-lvd1689m`(或类似)。首次下载前请查 [Meta DINOv3 官方仓库](https://github.com/facebookresearch/dinov3) 确认当前 ID。详见 [`docs/03-model-download.md`](./docs/03-model-download.md)。

### 3. 跑推理验证

```bash
python scripts/verify_dinov3.py
```

预期输出:
```
✅ DINOv3 推理链路贯通!
💡 提取到的特征向量维度: torch.Size([1, 1024])
⏱️ 单张图像推理耗时: ~XX ms
```

---

## 目录结构

```
poster-consistency-dinov3/
├── AGENTS.md              # 🚨 AI / 人类接手项目必读:硬规则 + 环境约束
├── README.md              # 本文件
├── chat.md                # 原始 Gemini 对话存档(历史参考,非真理)
├── .env.example           # 环境变量模板(HF_HOME、UV_CACHE_DIR 等)
├── .env                   # 实际配置(已 gitignore)
├── .gitignore
├── pyproject.toml         # uv 项目声明(锁定 PyTorch cu124 源)
├── uv.lock                # 依赖精确锁定,跨机复现
├── .venv/                 # Python 虚拟环境(已建)
│
├── docs/
│   ├── 01-business-context.md      # 业务背景、算法选型、Gemini 盲区分析
│   ├── 02-environment-setup.md     # 环境搭建 SOP
│   ├── 03-model-download.md        # 模型下载 SOP + 符号链接避坑
│   ├── 04-verification-script.md   # verify_dinov3.py 详解
│   ├── 05-roadmap.md               # 分阶段开发路线图
│   └── 06-pitfalls.md              # 已知坑与避雷清单
│
├── models/                # DINOv3 权重(gitignored)
├── data/                  # 业务数据(gitignored)
├── src/                   # 业务代码
├── scripts/               # 一次性脚本
├── .hf_cache/             # HF 缓存(gitignored,env 控制)
└── .uv_cache/             # uv 缓存(gitignored)
```

---

## 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| Python | 3.10.18 | uv 管理 |
| 包管理 | uv 0.7.17 | 不用 conda |
| 深度学习 | PyTorch 2.6.0 + CUDA 12.4 | 官方 wheel 内置 runtime |
| 视觉主干 | DINOv3-Large(300M) → 7B | Meta 自监督,Patch 级特征 |
| 微调 | peft (LoRA) + bitsandbytes (4-bit) | 7B 模型 4-bit 加载 <15GB 显存 |
| 图像 I/O | Pillow + OpenCV | OpenCV 负责 aspect-ratio padding |

---

## 当前进度

见 [`docs/05-roadmap.md`](./docs/05-roadmap.md)。快照:

- ✅ 环境:venv + PyTorch cu124 就绪
- ⏳ Phase 1:跑通 GAP 基线推理(进行中)
- ⬜ Phase 2:Patch-EMD 精细匹配
- ⬜ Phase 3:LoRA 微调
- ⬜ Phase 4:服务化部署

---

## 协作约定

- 中文沟通,代码注释/术语保留英文
- 任何写环境操作(`uv pip install`、下载 > 1GB)**先告知再执行**
- 磁盘剩余 < 20GB 立即停手,见 AGENTS.md R3

---

## 外部参考

- [Meta DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Hugging Face `facebook/`](https://huggingface.co/facebook)
- [uv 官方文档](https://docs.astral.sh/uv/)
- [PEFT/LoRA 文档](https://huggingface.co/docs/peft)
