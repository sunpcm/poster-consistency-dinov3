# 05 — 开发路线图(Roadmap)

> 分阶段交付,每个 Phase 有明确的 **Definition of Done**。
> 当前阶段:**Phase 1 进行中**(环境就绪,等装 HF 生态 + 下载模型)。

---

## Phase 0:基建(✅ 已完成)

- [x] 项目目录创建于 `/opt/workspace/yongli/poster-consistency-dinov3`
- [x] git 初始化
- [x] uv venv (Python 3.10.18)
- [x] PyTorch 2.6.0 + cu124 装好
- [x] AGENTS.md / README.md / docs/ 文档骨架就绪

---

## Phase 1:推理基线闭环(进行中)

**目标**:**任意一张 Logo 图 → DINOv3 → 1024 维特征向量 → 与参考向量算余弦相似度 → 输出得分**。
**度量方案**:GAP + cosine(基线,见 `01-business-context.md` §3)。

### 任务清单

- [ ] 1.1 安装 HF 生态:`uv pip install transformers accelerate peft huggingface_hub bitsandbytes opencv-python`(需用户许可)
- [ ] 1.2 核对 DINOv3-Large 在 HF 上的真实 model ID(见 `03-model-download.md` §1)
- [ ] 1.3 下载 DINOv3-Large 到 `models/dinov3-large/`
- [ ] 1.4 编写 `scripts/verify_dinov3.py`(模板见 `04-verification-script.md`)
- [ ] 1.5 跑通随机张量 smoke test
- [ ] 1.6 准备 3-5 张测试 Logo 图,跑真实图像推理
- [ ] 1.7 写 `src/similarity.py`:输入两张图,输出 cosine 得分
- [ ] 1.8 用 1 个参考 Logo + 5 张异常样本测试,记录得分分布

### Definition of Done(Phase 1)

- ✅ `python scripts/verify_dinov3.py` 在 5090 上跑通,延迟 < 100ms / 张
- ✅ 同一 Logo 不同角度照片的 cosine ≥ 0.85
- ✅ 不同 Logo 的 cosine ≤ 0.5
- ✅ **明确记录** GAP 在"漏印 / 缺角"等局部异常上的失效案例 → 作为 Phase 2 的动机证据

### Phase 1 不做的事

- ❌ 不优化推理速度(`torch.compile` 留到 Phase 2)
- ❌ 不做数据增强
- ❌ 不微调模型
- ❌ 不写 web 服务

---

## Phase 2:Patch-EMD 精细度量

**目标**:用最优传输替代 GAP,**捕捉局部��陷,产出异常热力图**。

### 任务清单

- [ ] 2.1 调研 OT 库选型:`pot`(Python Optimal Transport)/ `geomloss`(GPU Sinkhorn)
- [ ] 2.2 实现 `src/patch_emd.py`:输入两组 patch 特征 → Sinkhorn-OT 距离 + 传输矩阵
- [ ] 2.3 实现 aspect-ratio padding 预处理(替代默认 Resize),写入 `src/preprocess.py`
- [ ] 2.4 实现热力��:从 OT 传输矩阵反推每个 patch 的"未匹配代价" → 上采样到原图尺寸
- [ ] 2.5 在 Phase 1 的失效案例上验证 → 期望召回率显著提升
- [ ] 2.6 跑 `torch.compile(mode="reduce-overhead")` 优化吞吐
- [ ] 2.7 整理评估集,跑 P/R/F1

### Definition of Done(Phase 2)

- ✅ Phase 1 失效的"漏印"案例,Patch-EMD 能识别
- ✅ 热力图能定位异常区域(肉眼可看出在哪)
- ✅ 单张推理延迟 < 200ms(含 OT)
- ✅ 评估集上 Recall > 95%(漏检最关键)

---

## Phase 3:LoRA 微调(精度上限攻坚)

**目标**:针对业务数据微调 DINOv3,把"哪些视觉差异算异常、哪些算正常"显式教给模型。

### 任务清单

- [ ] 3.1 数据集:收集 ≥ 1000 对(reference, query, label)三元组
  - label = "same logo, normal" / "same logo, defective" / "different logo"
- [ ] 3.2 决定主干:Large + LoRA(快)还是 7B + LoRA + 4bit(精)?
  - 默认走 Large,7B 仅在 Large 撞天花板时启用
- [ ] 3.3 损失:**InfoNCE / SupCon 优先**,Triplet 仅作消融(见 `01-business-context.md` §4)
- [ ] 3.4 训练脚本:`src/train_lora.py`,基于 `peft` + `accelerate`
  - **强制 `save_total_limit=3`**(AGENTS.md R3)
- [ ] 3.5 评估前后差异
- [ ] 3.6 探索 Deformable Attention 模块作为 Phase 3.5 加分项

### Definition of Done(Phase 3)

- ✅ 微调后 F1 比 Phase 2 提升 ≥ 5%
- ✅ LoRA 权重 < 100MB,可独立分发
- ✅ 训练全过程磁盘剩余始终 > 30GB

---

## Phase 4:服务化与部署

**目标**:把推理链封装为 HTTP/gRPC 服务,接入业务流。

### 任务清单

- [ ] 4.1 选型:FastAPI(简单)vs Triton(高吞吐)
- [ ] 4.2 接口设计:批量上传,返回得分 + 热力图(base64)
- [ ] 4.3 Dockerfile(注意 5090 服务器无 mihomo,不要照搬 image2xml 的代理配置)
- [ ] 4.4 异常截断前置:用轻量 OpenCV 边缘检测过滤明显废片,省 GPU
- [ ] 4.5 监控:延迟、QPS、得分分布、漂移
- [ ] 4.6 灰度上线

### Definition of Done(Phase 4)

- ✅ 服务 P99 延迟 < 500ms
- ✅ 吞吐 ≥ 20 QPS(单 5090)
- ✅ 完整的运维文档:启停、扩容、回滚

---

## 不在路线图内 / 显式拒绝的方向

| 方向 | 拒绝理由 |
|---|---|
| 重新训练 DINOv3 主干 | 收益极低,工程量极大,无算力理由 |
| 接入 CLIP / SigLIP 替代 DINOv3 | DINOv3 在细粒度实例特征上更强,见 `01-business-context.md` §2 |
| 用 SD/ControlNet 做合成数据增强 | 合成形变 ≠ 真实布料形变,见 §4 |
| 多模态融合(图+文) | 业务不需要文本,徒增复杂度 |

---

## 决策回顾节点

每个 Phase 完成时,**回头检视一次**:
- 当前度量是不是真的在解决业务问题?(避免"为了优化指标而优化")
- 文档(尤其 01-business-context.md 的"决策日志")是否同步更新?
- 是否产生了新的事故 / 教训需要写进 AGENTS.md "事故记录"?
