# GFIN 系统架构文档

> 版本: v1 | 日期: 2026-02-23
> 基于全局上下文与成对融合特征的人-物交互检测

---

## 1. 项目概述

GFIN 是一个端到端的人-物交互 (HOI) 检测模型. 给定一张图像, 它:
1. 使用基于 DETR 的检测器检测目标 (人 + 物体)
2. 从检测到的候选框中构建人-物配对
3. 将每个配对分类为交互类别 (例如 "人骑自行车")

支持的基准数据集:
- **HICO-DET**: 600 个 HOI 类别 (117 个动词 x 80 个物体)
- **V-COCO**: 29 个动作类别

---

## 2. 高层架构

:::mermaid
graph TD
    A[输入图像] --> B[ResNet 骨干网络]
    B --> C[DETR Transformer 编码器-解码器]
    C --> D[目标候选: 边界框 + 置信度 + 特征]
    B --> E[GCE: 全局上下文编码器]
    D --> F[候选区域筛选: NMS + 置信度阈值]
    F --> G[HOF: 人-物融合模块]
    E --> H[全局记忆]
    G --> I[成对交互Token]
    H --> J[PID: 成对交互解码器]
    I --> J
    J --> K[HOI 分类: 动词 logits]
    K --> L[后处理: 先验分数 x logits]
    L --> M[输出: HOI 检测结果]
:::

---

## 3. 核心组件详解

### 3.1 DETR 目标检测器 (冻结)

**位置**: `detr/`

目标检测器基于 Facebook 的 DETR (DEtection TRansformer). 在 GFIN 训练过程中, 检测器被**冻结** -- 只训练交互头部.

| 组件 | 描述 | 输出形状 |
|------|------|---------|
| 骨干网络 | ResNet-50 或 ResNet-101 | 多尺度特征图 (C2-C5) |
| Transformer 编码器 | 6 层, 8 个注意力头 | Memory: (HW, B, 256) |
| Transformer 解码器 | 6 层, 100 个查询 | Queries: (100, B, 256) |
| 预测头 | 分类 + 边界框头 | logits: (B, 100, 81), boxes: (B, 100, 4) |

**级联预测**: GFIN 使用 DETR 解码器的级联层输出. 从 `cascade_layer` 到 `dec_layers` 的预测结果被聚合, 允许模型使用多层解码器输出作为候选区域, 提升召回率.

### 3.2 候选区域准备

**位置**: `gfin.py` -> `GFIN.prepare_region_proposals()`

对 DETR 输出进行后处理, 筛选高质量候选:

1. **NMS 过滤** (batched_nms, `fg_iou_thresh=0.5`)
2. **置信度阈值过滤** (`box_score_thresh=0.2`)
3. **实例数量平衡**: 确保人和物体各有 `min_instances` (4) 到 `max_instances` (15) 个
4. **输出**: 每张图像的字典, 包含 boxes, scores, labels, hidden_states, location

### 3.3 GCE: 全局上下文编码器

**位置**: `gfin_module/gce.py`, 在 `interaction_head.py` -> `GEC_forward()` 中调用

从骨干网络特征编码图像级全局上下文.

:::mermaid
graph TD
    A[ResNet C5 特征: B x 2048 x H x W] --> B[1x1 卷积投影: B x 256 x H x W]
    B --> C[Transformer 编码器: enc_inter_layers 层]
    C --> D[残差连接: + DETR 编码器特征]
    D --> E[拼接 DETR memory 投影]
    E --> F[全局记忆: HW x B x 512]
:::

| 参数 | 默认值 | 描述 |
|-----|--------|------|
| enc_inter_layers | 4 | 编码器层数 |
| hidden_dim | 256 | 特征维度 |
| nheads | 8 | 注意力头数 |

### 3.4 HOF: 人-物融合模块

**位置**: `gfin_module/hof.py`, 在 `interaction_head.py` forward() 中调用

通过自定义的多头注意力机制, 融合外观特征与空间关系信息.

:::mermaid
graph TD
    A[一元Token: N x 256] --> B[线性投影 Q/K/V]
    C[空间编码: NxN x 36] --> D[空间 MLP: 36 -> 512]
    D --> E[空间嵌入: N x N x 512]
    B --> F[多头成对注意力]
    E --> F
    F --> G[注意力权重: sigmoid 门控]
    G --> H[消息聚合]
    H --> I[LayerNorm + FFN]
    I --> J[增强 Token: N x 256]
:::

**空间编码** (每对框 36 维向量):
- 相对中心坐标 (归一化)
- 相对宽高
- 相对面积
- 宽高比
- 框间 IoU
- 相对距离与方向
- 上述特征的对数变换

HOF 模块使用 **3 个并行注意力分支**, 采用 sigmoid 门控 (而非 softmax). 这允许每个物体独立关注多个相关实体, 而不是在所有实体间归一化注意力 -- 这对包含多个并发交互的场景很重要.

### 3.5 PID: 成对交互解码器

**位置**: `gfin_module/pid.py`, 在 `interaction_head.py` forward() 中调用

交叉注意力解码器, 通过关注全局上下文记忆生成 HOI 预测.

:::mermaid
graph TD
    A[成对Token: 拼接人+物特征: P x 1 x 512] --> B[自注意力]
    C[GCE 全局记忆: HW x 1 x 512] --> D[交叉注意力]
    B --> D
    D --> E[FFN]
    E --> F[dec_inter_layers 次迭代]
    F --> G[HOI Token: layers x P x 512]
    G --> H[线性分类器: 512 -> num_classes]
    H --> I[每对动词 logits]
:::

| 参数 | 默认值 | 描述 |
|-----|--------|------|
| dec_inter_layers | 4 | 解码器层数 |
| d_model | 512 (256*2) | 输入维度 (人+物特征拼接) |
| return_intermediate | True | 返回所有层输出用于辅助损失 |

### 3.6 损失函数

**位置**: `ops.py` -> `binary_focal_loss()`, 在 `gfin.py` -> `compute_interaction_loss()` 中调用

- **Binary Focal Loss**, alpha=0.5, gamma=0.2
- 先验分数: human_score * object_score (逐元素乘)
- 最终分数 = sigmoid(logits) * prior
- 损失在所有解码器层上平均 (辅助监督)
- 分布式: 损失按全局正样本数归一化

### 3.7 后处理

**位置**: `gfin.py` -> `GFIN.postprocessing()`

- 分数 = sigmoid(logits_last_layer) * (human_score^2.8 * object_score^2.8)
- 推理时使用幂指数 `p=2.8` (训练时 `p=1.0`)
- 可选: 成对 NMS (三元组 NMS), 通过 `--pnms` 开启

---

## 4. 数据流

### 4.1 训练流水线

:::mermaid
graph TD
    A[DataFactory: HICODet/VCOCO] --> B[DataLoader + DistributedSampler]
    B --> C[GFIN.forward: training=True]
    C --> D[DETR 前向: 冻结, no_grad]
    D --> E[级联层处理]
    E --> F[候选区域准备]
    F --> G[InteractionHead 前向]
    G --> H[GCE -> HOF -> PID]
    H --> I[compute_interaction_loss]
    I --> J[Binary Focal Loss]
    J --> K[反向传播 + 梯度裁剪 + 优化器步进]
    K --> L[CustomisedDLE 引擎循环]
:::

### 4.2 推理流水线

:::mermaid
graph TD
    A[图像输入] --> B[GFIN.forward: training=False]
    B --> C[DETR 候选框]
    C --> D[InteractionHead]
    D --> E[后处理]
    E --> F[每张图输出字典]
    F --> G1[boxes + pairing 索引]
    F --> G2[scores + labels]
    F --> G3[attention maps]
:::

---

## 5. 模块依赖关系

:::mermaid
graph TD
    main[main.py] --> gfin_mod[gfin.py]
    main --> utils[utils.py]
    gfin_mod --> interaction[interaction_head.py]
    gfin_mod --> ops[ops.py]
    gfin_mod --> detr_models[detr/models/]
    interaction --> base[gfin_module/base_interaction_head.py]
    interaction --> hof[gfin_module/hof.py]
    base --> gce[gfin_module/gce.py]
    base --> pid[gfin_module/pid.py]
    hof --> mbf[gfin_module/mbf.py]
    utils --> pocket_lib[pocket/]
    utils --> hicodet_ds[hicodet/hicodet.py]
    utils --> vcoco_ds[vcoco/vcoco.py]
    inference[inference.py] --> gfin_mod
    inference --> add_on[add_on/infer_util.py]
:::

---

## 6. 目录结构

```
GFIN/
+-- main.py                     # 训练/评估入口
+-- inference.py                # 推理/可视化入口
+-- gfin.py                     # GFIN 检测器构建 + 主模型类
+-- interaction_head.py         # InteractionHead (GCE + HOF + PID 编排)
+-- ops.py                      # 损失函数, 空间编码, 匹配器
+-- utils.py                    # 数据工厂, 训练引擎, 评估工具
+-- upt.py                      # UPT 变体检测器 (替代方案)
+-- upt_interaction_head.py     # UPT 交互头 (替代方案)
+-- vcoco_test.py               # V-COCO 独立评估脚本
+--
+-- gfin_module/                # GFIN 核心组件
|   +-- activations.py           # 共享激活函数工具
|   +-- base_interaction_head.py # 基类: 编码/解码辅助方法
|   +-- gce.py                   # 全局上下文编码器 (TransformerEncoder)
|   +-- hof.py                   # 人-物融合模块
|   +-- pid.py                   # 成对交互解码器 (TransformerDecoder)
|   +-- mbf.py                   # 多分支融合
|   +-- position_encoding.py     # 位置编码工具
|   +-- coop.py                  # 协作学习模块
+--
+-- detr/                       # DETR 目标检测器 (HOI 训练时冻结)
|   +-- models/                  # 骨干网络, transformer, 预测头
|   +-- datasets/                # 数据变换
|   +-- util/                    # 框操作, 杂项工具
+--
+-- pocket/                     # 轻量级深度学习库 (fredzzhang/pocket)
|   +-- core/                    # 训练引擎, 分布式
|   +-- data/                    # 采样器, 数据集类
|   +-- models/                  # 模型构建模块
|   +-- ops/                     # 框操作, 变换
|   +-- utils/                   # 计量器, 关联, 可视化
+--
+-- add_on/                     # 推理与可视化辅助工具
|   +-- infer_util.py            # 图像/视频推理运行器
|   +-- pnms.py                  # 成对/三元组 NMS
|   +-- draw_pic.py              # 可视化绘制
|   +-- attention.py             # 注意力可视化
|   +-- hook_util.py             # 前向 hook 工具
|   +-- net.py                   # 网络辅助模块
|   +-- nms.py                   # NMS 工具
|   +-- tsne_vis.py              # t-SNE 可视化
+--
+-- hicodet/                    # HICO-DET 数据集工具
+-- vcoco/                      # V-COCO 数据集工具
+-- experiments/                # 训练/评估 shell 脚本
+-- scripts/                    # Docker 辅助脚本
+-- docs/                       # 项目文档
+-- logs/                       # 训练输出目录
```

---

## 7. 关键超参数

| 参数 | HICO-DET | V-COCO | 描述 |
|-----|----------|--------|------|
| backbone | resnet101 | resnet50 | CNN 骨干网络 |
| num_classes | 117 | 29 | 动词/动作类别数 |
| num_obj_classes | 81 | 81 | COCO 物体类别数 |
| enc_inter_layers | 4 | 4 | GCE 编码器层数 |
| menc_inter_layer | 4 | 4 | HOF 层数 |
| dec_inter_layers | 4 | 4 | PID 解码器层数 |
| hidden_dim | 256 | 256 | DETR/GCE 特征维度 |
| repr_dim | 512 | 512 | 表示维度 |
| lr_head | 2e-4 | 2e-4 | 交互头学习率 |
| batch_size | 8 | 8 | 每 GPU batch 大小 |
| epochs | 20 | 20 | 训练轮数 |
| lr_drop | 10 | 10 | 学习率衰减轮数 |
| alpha | 0.5 | 0.5 | Focal Loss alpha |
| gamma | 0.2 | 0.2 | Focal Loss gamma |

---

## 8. 设计决策与权衡

### 8.1 冻结检测器
DETR 检测器在交互头训练期间被冻结. 这减少了 GPU 显存使用和训练时间, 但意味着检测器无法适应 HOI 特定的特征.

### 8.2 级联层预测
使用来自多个解码器层 (cascade_layer 到 dec_layers) 的预测增加了候选区域池, 提升召回率, 代价是交互头计算量增大.

### 8.3 HOF 中使用 Sigmoid 而非 Softmax
HOF 使用基于 sigmoid 的门控代替 softmax 注意力. 这允许每个物体独立关注多个实体, 而不是在所有实体间归一化注意力 -- 这对包含多个并发交互的场景尤为重要.

### 8.4 辅助解码器损失
所有 PID 解码器层的输出都参与损失计算 (取平均), 为中间层提供梯度信号, 改善收敛.
