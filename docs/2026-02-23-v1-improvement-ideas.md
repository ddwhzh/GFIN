# GFIN 改进方向与路线图

> 版本: v1 | 日期: 2026-02-23
> 基于 [HOI-Learning-List](https://github.com/DirtyHarryLYL/HOI-Learning-List) 及最新研究趋势

---

## 当前 GFIN 架构总结

GFIN (Neural Networks 2023) 使用 冻结 DETR + GCE + HOF + PID 的流水线架构. 核心优势在于:
- 全局上下文编码 (GCE) 提供图像级语义
- 成对融合 (HOF) 的 sigmoid 门控多头注意力
- 级联层预测提升候选框召回率

---

## 改进方向

### 方向 1: CLIP/视觉-语言模型集成 [高优先级]

**动机**: 当前 GFIN 仅使用视觉特征, 缺乏语义先验. 2023-2025 年的 SOTA 方法 (HOICLIP, CLIP4HOI, UniHOI, EZ-HOI, HOLa) 都利用 CLIP 特征获得了显著提升.

**具体方案**:

#### 1a. CLIP 文本先验作为分类器初始化
- 将 `box_pair_predictor` (当前为 `nn.Linear(512, num_classes)`) 的权重用 CLIP 文本编码器的动词/动作文本嵌入初始化
- 文本模板: "a person {verb}ing a {object}" -> CLIP text encoder -> 作为分类器权重
- 参考: GEN-VLKT (CVPR 2022), HOICLIP (CVPR 2023)
- **实现难度**: 低
- **预期收益**: mAP +1~3%

```python
# 伪代码示例
import clip
model, preprocess = clip.load("ViT-B/32")
verb_texts = [f"a person {v}ing a {obj}" for v in verbs for obj in objects]
text_features = model.encode_text(clip.tokenize(verb_texts))
predictor.weight.data = text_features  # 初始化分类器
```

#### 1b. CLIP 视觉特征增强
- 使用 CLIP 视觉编码器提取全局图像特征
- 与 GCE 的全局记忆做特征融合 (拼接或多分支融合)
- 参考: ADA-CM (ICCV 2023), ISA-HOI (2024)
- **实现难度**: 中
- **预期收益**: mAP +2~4%

#### 1c. 多层知识蒸馏
- 使用 CLIP 作为教师模型进行多层蒸馏: 全局图像级 + 局部 union region + 实例级
- 参考: MLKD (WACV 2024)
- **实现难度**: 中高
- **预期收益**: mAP +2~5%

---

### 方向 2: "Toxic Siblings" 去偏 [高优先级]

**动机**: ICCV 2025 的 "No More Sibling Rivalry" 发现 HOI Transformer 中的 "Toxic Siblings" 偏差问题 -- 同一人-物对的多个交互标签之间存在竞争, 导致模型对共现交互的预测相互抑制. 这在 HICO-DET 上取得了 +9.18% 的 baseline 提升.

**具体方案**:
- 对比-再校准 (Contrastive-then-Calibration): 在训练时添加对比损失, 鼓励模型区分不同交互
- 合并-再拆分 (Merge-then-Split): 先将同一对的多个交互预测合并, 再拆分为独立预测
- **改动位置**: 修改 `compute_interaction_loss` 和后处理逻辑
- **实现难度**: 中
- **预期收益**: mAP +3~5%

---

### 方向 3: 人体姿态特征融合 [中优先级]

**动机**: 人体姿态是判断交互类型的关键线索 (如 "踢" vs "拿"). GFIN 当前仅使用边界框级别的空间编码, 缺失细粒度的姿态信息.

**具体方案**:

#### 3a. 关键点空间编码增强
- 扩展 `compute_spatial_encodings` 中的 36 维空间编码, 增加人体关键点到物体的相对位置特征
- 使用 DETR / ViTPose 提取关键点
- 参考: Pose-Aware (CVPR 2024), STIP (CVPR 2022)
- **改动位置**: `ops.py` 的 `compute_spatial_encodings` + HOF 输入
- **实现难度**: 中
- **预期收益**: mAP +1~3%

#### 3b. 人体部位图 (Body Part Map)
- 将人体区域分割为部位区域, 提取部位级特征
- 参考: PartMap (ECCV 2022)
- **实现难度**: 高
- **预期收益**: mAP +2~4%

---

### 方向 4: 解码器架构升级 [中优先级]

**动机**: GFIN 的 PID 使用标准 Transformer Decoder, 而近年来的解码器架构有了显著改进.

**具体方案**:

#### 4a. DAB-DETR 风格的锚框查询
- 将 PID 的查询从可学习嵌入改为基于人-物对位置的动态锚框
- 参考: DAB-DETR, DN-DETR
- **实现难度**: 中高
- **预期收益**: 收敛速度提升, mAP +1~2%

#### 4b. 可变形注意力 (Deformable Attention)
- 用可变形注意力替换 PID 中的标准交叉注意力
- 减少计算量, 同时聚焦关键区域
- DETR 子模块中已有可变形注意力的 CUDA 实现 (`detr/models/necks/module/ops/`)
- **实现难度**: 中
- **预期收益**: 推理速度提升 + mAP 持平或小幅提升

#### 4c. 多尺度特征
- 当前 GCE 仅使用 C5 (最后一层) 特征
- 引入 FPN 或多尺度注意力, 融合 C3-C5 特征
- 参考: MUREN (CVPR 2023), Deformable DETR
- **实现难度**: 中
- **预期收益**: 小目标检测改善, mAP +1~2%

---

### 方向 5: 训练策略优化 [中优先级]

**具体方案**:

#### 5a. 联合训练检测器
- 当前 DETR 检测器完全冻结, 尝试在后期 fine-tune 骨干网络 (如最后一个 stage)
- 添加 `--unfreeze-backbone-epoch` 参数, 在训练后期解冻
- **实现难度**: 低
- **预期收益**: mAP +0.5~1.5%

#### 5b. 标签平滑与混合精度
- 对稀有类使用标签平滑或过采样
- 使用 AMP (Automatic Mixed Precision) 加速训练
- **实现难度**: 低
- **预期收益**: 训练速度 2x, 稀有类 mAP 提升

#### 5c. 动态标签分配
- 替换当前的固定 IoU 阈值 (0.5) 标签分配
- 使用 SimOTA 或匈牙利匹配 (已有 `HungarianMatcher` 实现但未默认启用)
- **实现难度**: 低 (已有代码基础)
- **预期收益**: mAP +0.5~2%

---

### 方向 6: 数据增强与合成 [低优先级]

**具体方案**:
- 使用扩散模型生成 HOI 训练数据
- 参考: DiffHOI, SynHOI, DiffusionHOI (NeurIPS 2024)
- **实现难度**: 高
- **预期收益**: 稀有类 mAP 显著提升

---

### 方向 7: 图结构建模 [低优先级]

**动机**: MGNM (2025) 表明 GNN 在显式建模 HOI 关系结构方面具有优势.

**具体方案**:
- 在 HOF 之后增加图卷积层, 建模人-物-人之间的高阶关系
- 参考: SCG (ICCV 2021), SGCN4HOI, MGNM (2025)
- **实现难度**: 中
- **预期收益**: 多人场景 mAP 提升

---

## 推荐实施路线

### Phase 1: 低成本高收益 (1-2 周)
1. **CLIP 文本先验** (方向 1a) -- 最小改动, 预期收益高
2. **启用匈牙利匹配** (方向 5c) -- 代码已有基础
3. **解冻骨干网络后期 fine-tune** (方向 5a) -- 仅需加参数

### Phase 2: 核心架构升级 (2-4 周)
4. **Toxic Siblings 去偏** (方向 2) -- 改动集中在 loss
5. **CLIP 视觉特征增强** (方向 1b) -- 与 GCE 融合
6. **关键点空间编码** (方向 3a) -- 扩展已有空间编码

### Phase 3: 深度优化 (4-8 周)
7. **多尺度 GCE** (方向 4c)
8. **可变形注意力 PID** (方向 4b)
9. **多层知识蒸馏** (方向 1c)

---

## 参考文献

| 方法 | 会议/期刊 | HICO-DET mAP | 核心贡献 |
|------|----------|-------------|---------|
| GFIN | NN 2023 | 见论文 | 全局上下文 + 成对融合 |
| HOICLIP | CVPR 2023 | 34.69 | CLIP 知识迁移 |
| PViC | ICCV 2023 | 34.69 | 先验引导可视化 |
| ViPLO | CVPR 2023 | 37.22 | 姿态引导 |
| MUREN | CVPR 2023 | 32.87 | 多尺度关系 |
| GEN-VLKT | CVPR 2022 | 33.75 | VL 知识迁移 |
| UPT | CVPR 2022 | 32.62 | 一元-成对 Transformer |
| Merge-then-Split | ICCV 2025 | +9.18% | 去偏 |
| EZ-HOI | NeurIPS 2024 | - | 高效零样本 |
| DiffusionHOI | NeurIPS 2024 | - | 扩散生成增强 |
