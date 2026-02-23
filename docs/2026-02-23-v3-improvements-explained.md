# GFIN v3 原创改进详解

> 版本: v3 | 日期: 2026-02-23
> 本文用图和例子解释每个改进, 确保你能理解"为什么这样做"和"具体改了什么"

---

## 0. 全局概览: 改了哪里?

先回顾 GFIN 的流水线, 然后标注每个改进的作用位置:

```
图像
  |
  v
+------------------+
| DETR (冻结)       |  --> 检测框 + 特征 + 置信度
+------------------+
  |
  v
+------------------+     +----------------------------+
| GCE 全局上下文    | --> | 全局记忆 (HW, B, 512)      |
+------------------+     +----------------------------+
  |                                    |
  v                                    |
+------------------+                   |
| HOF 成对融合      | <-- [A] 有向注意力 + [D] 空间精炼
+------------------+
  |
  v
+------------------+
| PID 交互解码器    | --> HOI 特征 (N_pairs, 512)
+------------------+
  |
  +-----> [B] 逐类先验校准 (修改 prior score 计算)
  |
  +-----> 主分类器 --> 主损失 (focal loss)
  |
  +-----> [C] 原型记忆 --> 辅助损失
  |
  +-----> [E] 粗粒度分类 --> 层级损失
  |
  +-----> [F] 注意力蒸馏 --> 蒸馏损失
```

| 改进 | 位置 | 一句话 |
|-----|------|-------|
| A | HOF 内部 | 让三个注意力分支"看不同的东西" |
| B | prior score | 每个动作类别学自己的分数放大系数 |
| C | 损失函数 | 维护每个类别的"标准答案"特征 |
| D | HOF 内部 | 让每层自动选择关注哪些空间关系 |
| E | 损失函数 | 先学粗分类, 再学细分类 |
| F | 损失函数 | 用框的重叠度引导注意力该看哪里 |

---

## A. 有向 HOF 注意力 (`--directional-hof`)

### 问题: 三个分支做一样的事

原始 HOF 有三个注意力分支 (attn1, attn2, attn3), 但它们接收**完全相同**的输入:

```
原始: 三个分支 ALL 收到 [K_人, K_物, 空间特征]

Branch 1: Linear([K_人, K_物, 空间]) --> sigmoid --> weight1
Branch 2: Linear([K_人, K_物, 空间]) --> sigmoid --> weight2
Branch 3: Linear([K_人, K_物, 空间]) --> sigmoid --> weight3

最终权重 = (weight1 + weight2 + weight3) / 3    <-- 简单平均
```

三个分支的唯一区别是随机初始化的权重不同. 这相当于花三倍的参数做几乎一样的事.

### 解决: 每个分支看不同的东西

```
改进: 三个分支看不同的"视角"

Branch 1 (前向):  [K_人, K_物, 空间]     "人对物的注意力"
Branch 2 (反向):  [K_物, K_人, 空间]     "物对人的约束"  (顺序反了!)
Branch 3 (交互):  [K_人*K_物, |K_人-K_物|, 空间]  "两者的相似/差异"

最终权重 = g1*w1 + g2*w2 + g3*w3          <-- 学习的门控
```

### 具体例子

场景: 图中有一个**人**和一个**自行车**

```
                   Branch 1 (前向)         Branch 2 (反向)         Branch 3 (交互)
                   "人怎么看自行车"         "自行车怎么约束人"       "两者有多相似/不同"
输入:              [人特征, 车特征, 距离]   [车特征, 人特征, 距离]   [人*车, |人-车|, 距离]
关注的:            人的手在车把附近?        车的座位在人下方?        两者尺度差异大吗?
适合判断:          "hold", "push"          "ride", "sit_on"        "carry" vs "ride"
```

**门控机制**: 不同实例会自动选择不同分支的权重

```
人骑自行车:  gate = [0.2, 0.6, 0.2]  --> 主要用 Branch 2 (物对人的位置约束很重要)
人扔飞盘:    gate = [0.5, 0.2, 0.3]  --> 主要用 Branch 1 (人手部的动作方向很重要)
```

### 代码位置

`gfin_module/hof.py` -> `HOFmoduleLayer.forward()`:

```python
# 三个不同视角的特征
fwd_features = torch.cat([Kappe[:, i], Kappe[:, j], Qspat], dim=-1)      # 前向
bwd_features = torch.cat([Kappe[:, j], Kappe[:, i], Qspat], dim=-1)      # 反向 (注意 i,j 反了)
int_features = torch.cat([Kappe[:, i] * Kappe[:, j],                      # 交互
                          (Kappe[:, i] - Kappe[:, j]).abs(), Qspat], dim=-1)

# 学习的门控 (每个节点自己决定用哪个分支多一些)
gate = softmax(direction_gate(appearance), dim=-1)   # (N, 3)
weights = g1 * w1 + g2 * w2 + g3 * w3               # 加权组合
```

---

## B. 可学习逐类先验校准 (`--learnable-prior`)

### 问题: 所有类别用同一个"魔法数字"

GFIN 用检测框的置信度来过滤人-物配对. 原始代码做了一件事:

```
训练时:  prior = score^1.0    (不放大, 检测分数直接用)
推理时:  prior = score^2.8    (放大差距, 抑制低置信检测)
                 ^^^^^
                 这个 2.8 是手动调出来的, 对所有 117 个动作类别一视同仁
```

### 为什么这有问题?

```
以 HICO-DET 为例:

类别 "ride bicycle":     训练样本 1000+, 检测器对自行车识别很准 --> score 通常 > 0.8
类别 "lasso elephant":   训练样本 < 10,  检测器对大象不太准     --> score 通常 0.3~0.6

如果都用 p=2.8:
  0.8^2.8 = 0.49   (还行)
  0.5^2.8 = 0.12   (几乎被过滤掉了!)

稀有类的检测分数本来就低, 再用 2.8 次方一放大, 直接被干掉了.
```

### 解决: 每个类别学自己的 p

```
prior_calibration = nn.Parameter(ones(117))   # 117个可学习参数, 初始化为1.0

ride_bicycle:     p[12] = 学到 1.8   (常见类, 可以适度放大)
lasso_elephant:   p[95] = 学到 0.5   (稀有类, 几乎不放大, 保留弱检测)
sit_on_chair:     p[42] = 学到 2.5   (椅子检测很准, 可以大力放大)
```

### 代码位置

`interaction_head.py` -> `compute_prior_scores()`:

```python
# 原始: 所有类别同一个 p
p = 1.0 if self.training else 2.8
s_h = scores[x_keep].pow(p)              # 标量指数

# 改进: 每个类别独立的 p
p = self.prior_calibration.clamp(min=0.1)           # (K,) 向量
s_h = scores[x_keep].unsqueeze(-1).pow(p.unsqueeze(0))  # (M, K) 矩阵
#     每个配对 M x 每个类别 K 都有不同的放大系数
```

---

## C. 稀有动作原型记忆库 (`--prototype-memory`)

### 问题: 稀有类见得少, 学不好

```
HICO-DET 的类别分布极度不均衡:

hold umbrella:     ████████████████████  2000+ 样本
ride bicycle:      ████████████████      1500+ 样本
lasso cow:         █                     5 样本
stab cake:         █                     3 样本
                                         ^^^^^^^^^
                                         模型几乎学不到稳定的特征表示
```

### 解决: 为每个类别维护一个"标准答案"

```
原型记忆库 = (117, 512) 矩阵
每一行是一个类别的"标准特征向量"

                  训练过程:
Batch 1:  出现了 "ride bicycle" --> 更新 prototypes[12] (EMA平滑)
Batch 2:  出现了 "hold umbrella" --> 更新 prototypes[0]
Batch 3:  出现了 "ride bicycle" --> 再次更新 prototypes[12]
...
Batch 99: 出现了 "lasso cow"    --> 更新 prototypes[95]  (终于出现了!)
Batch 100-200: prototypes[95] 保持不变 (因为再也没出现)
              但这个原型仍然可以作为"参考答案"提供辅助分类
```

### 工作方式

```
EMA 更新公式:
  prototype_new = 0.999 * prototype_old + 0.001 * 当前batch均值

辅助分类:
  当前 HOI 特征 --> 与所有 117 个原型做余弦相似度 --> 辅助 focal loss

直觉: "这个人-物配对的特征和哪个类别的标准答案最像?"
```

### 数据流图

```
HOI 特征 (N, 512)
  |
  |--- 主路径 ---> box_pair_predictor ---> 主 focal loss
  |
  |--- 辅助路径:
  |    cosine_similarity(特征, prototypes) --> (N, 117) 相似度分数
  |    focal_loss(相似度, 标签) --> proto_loss * 0.1
  |
  |--- 更新路径 (detach, 不传梯度):
       对出现的类别: prototype = 0.999*old + 0.001*new
```

### 代码位置

`gfin.py` -> `ActionPrototypeMemory` 类 + `compute_interaction_loss()`:

```python
# EMA 更新 (不传梯度)
self.action_prototypes.update(hoi_features.detach(), labels)

# 辅助分类
proto_sim = self.action_prototypes.similarity(hoi_features)  # cosine sim
proto_scores = (proto_sim + 1.0) / 2.0                       # [-1,1] -> [0,1]
proto_loss = binary_focal_loss(proto_scores, labels)
loss += 0.1 * proto_loss
```

---

## D. 空间编码迭代精炼 (`--spatial-refine`)

### 问题: 空间特征在所有层都一样

HOF 有 4 层 (默认), 每层都使用**完全相同**的空间特征:

```
空间特征 = 36维向量, 包含:
  [人中心x, 人中心y, 物中心x, 物中心y,    位置
   人宽, 人高, 物宽, 物高,                尺寸
   人面积, 物面积, 面积比,                面积
   人宽高比, 物宽高比,                    形状
   IoU,                                   重叠度
   左/右/上/下方向距离,                    方向
   以上18维的 log 变换]                    对数特征

HOF Layer 1: 用这 36 维 --> 投影到 512 维
HOF Layer 2: 用同样的 512 维
HOF Layer 3: 用同样的 512 维
HOF Layer 4: 用同样的 512 维
             ^^^^^^^^^^^^
             每层都看一样的空间信息, 没有层级递进
```

### 解决: 每层根据当前理解动态调整空间关注点

```
HOF Layer 1:
  当前外观理解: "这是人和自行车"
  空间门控 = sigmoid(Linear(外观))
  --> 打开: 位置, IoU, 方向  (先看粗粒度)
  --> 关闭: 面积比, 宽高比   (暂时不需要)

HOF Layer 2:
  当前外观理解: "人在自行车上方"
  空间门控 = sigmoid(Linear(更新后的外观))
  --> 打开: 面积比, IoU      (开始看细节)
  --> 关闭: 粗粒度方向        (已经知道了)

HOF Layer 3-4:
  外观越来越精细 --> 空间注意力也越来越聚焦接触区域
```

### 代码原理

```python
# 每个 HOF Layer 内部:
sg = sigmoid(Linear(appearance))    # (N, 512)  门控向量
sg = reshape(sg)                    # (8, N, 64) 多头
Qspat = Qspat * sg.unsqueeze(2)    # (8, N, N, 64) 逐元素调制

# unsqueeze(2) 让门控对"目标维度"广播:
# sg:    (8, N, 1, 64)   每个源节点的门控
# Qspat: (8, N, N, 64)   所有源-目标对的空间特征
# 结果:  (8, N, N, 64)   源节点i关注的空间维度被选择性放大/抑制
```

---

## E. 动词层级分类 (`--hierarchical-verb`)

### 问题: 117 个细粒度动词太难直接学

```
HICO-DET 的 117 个动词对于模型来说很难区分:

  kick  vs  hit     (都是接触类, 视觉上很像)
  ride  vs  sit_on  (都是位置类, 区别在于是否运动)
  eat   vs  sip     (都是消费类, 区别很微妙)

当模型判断错 "kick" 还是 "hit" 时, 梯度信号说"完全错了" -- 但实际上它已经学到了
这是一种"接触动作", 只是细节没搞对. 这个部分正确的信息完全浪费了.
```

### 解决: 加一个粗粒度分类头, 先学大类

```
117 个细粒度动词 --> 自动归类为 6 个粗粒度组:

+-- contact (接触): hold, carry, hug, kick, hit, throw, catch...
+-- tool_use (工具): cut_with, brush_with, wield, stir, pour...
+-- positional (位置): ride, sit_on, lie_on, stand_on, jump...
+-- manipulation (操作): open, repair, cook, wash, clean, load...
+-- observation (观察): watch, read, eat, drink, point, greet...
+-- none (无交互): no_interaction
```

### 工作方式

```
HOI 特征 (N, 512)
  |
  +--> 主分类器: Linear(512, 117) --> 细粒度预测 --> 主 focal loss
  |
  +--> 粗分类器: Linear(512, 6)  --> 粗粒度预测 --> 层级 focal loss (* 0.3)
                                       ^
                                       |
                            粗标签从细标签自动聚合:
                            kick=1, hit=0 --> contact=1
                            ride=0, sit=0 --> positional=0
```

### 具体例子

```
场景: 人踢足球

细粒度真值: kick=1, 其他=0
粗粒度真值: contact=1, 其他=0  (因为 kick 属于 contact 组)

模型预测:
  细粒度: kick=0.3, hit=0.5, throw=0.1    (预测错了, 选了 hit)
  粗粒度: contact=0.9, tool=0.05           (但粗粒度对了!)

梯度反馈:
  主损失: "kick 错了!" (penalty)
  层级损失: "contact 对了!" (reward)
  --> 模型知道方向是对的, 只需在 contact 组内继续细化
```

---

## F. 注意力蒸馏 (`--attn-distill`)

### 问题: HOF 注意力的学习没有直接监督

```
HOF 的注意力权重决定了"每个物体该关注谁":

  当前物体列表: [人A, 人B, 自行车, 狗]

  注意力矩阵 (N x N):
              人A    人B    自行车   狗
  人A     [  --    0.3    ???    ???  ]
  人B     [ 0.2    --    ???    ???  ]
  自行车  [ ???   ???     --    ???  ]
  狗      [ ???   ???    ???     --  ]

  ??? 的值完全靠反向传播从最终的分类损失间接学到,
  没有任何直接信号告诉模型"自行车应该更关注人A而不是狗"
```

### 解决: 用框的 IoU 作为"该看谁"的提示

```
直觉: 空间上越近/越重叠的物体, 越可能发生交互, 注意力应该越高

IoU 矩阵 (计算所有框两两之间的重叠度):
              人A    人B    自行车   狗
  人A     [  1.0   0.0    0.6    0.1  ]
  人B     [  0.0   1.0    0.0    0.7  ]
  自行车  [  0.6   0.0    1.0    0.0  ]
  狗      [  0.1   0.7    0.0    1.0  ]

人A 和自行车 IoU=0.6 (重叠多, 可能在骑车)
人B 和狗    IoU=0.7 (重叠多, 可能在遛狗)
人A 和狗    IoU=0.1 (几乎不重叠, 可能没交互)

蒸馏损失: MSE(注意力矩阵, IoU矩阵)
让注意力权重逼近 IoU 分布
```

### 为什么是"弱监督"而不是"强监督"?

```
IoU 不完美:
  - "人看电视": 人和电视可能 IoU=0 (不重叠), 但确实在交互
  - "人抱猫":   人和猫 IoU 很高, 确实在交互

所以我们:
  1. 只用 MSE loss (不用 KL divergence, 更宽松)
  2. 权重很小 (0.1, 只是一个"提示"而不是"命令")
  3. 仅作为辅助损失, 主分类损失仍然主导学习
```

### 代码位置

`gfin.py` -> `compute_interaction_loss()`:

```python
# 计算 IoU 矩阵作为蒸馏目标
iou_target = box_iou(bx, bx)                         # (N, N) 框两两 IoU

# 取 HOF 多头平均注意力
avg_attn = stack([w.squeeze(-1) for w in hof_attn]).mean(0)  # (N, N)

# MSE 损失: 让注意力分布逼近 IoU 分布
distill_loss += mse_loss(avg_attn, iou_target)
loss += 0.1 * distill_loss
```

---

## 总结: 损失函数全景

启用全部改进后, 训练时的总损失由以下部分组成:

```
Total Loss
  |
  +-- 主分类损失 (focal loss)                    权重: 1.0
  |     每层 PID 解码器输出 * prior --> focal loss, 取平均
  |
  +-- [v2] 去偏损失 (sibling contrastive)        权重: 0.5
  |     鼓励正标签分数 > 负标签分数 + margin
  |
  +-- [A/D] 无额外损失, 改变的是 HOF 内部计算
  |
  +-- [B] 无额外损失, prior_calibration 通过主损失梯度学习
  |
  +-- [C] 原型辅助损失 (focal loss)              权重: 0.1
  |     cosine_sim(特征, 原型) vs 真实标签
  |
  +-- [E] 层级辅助损失 (focal loss)              权重: 0.3
  |     粗粒度预测 vs 聚合粗标签
  |
  +-- [F] 注意力蒸馏损失 (MSE)                   权重: 0.1
        HOF 注意力 vs 框 IoU 矩阵
```

---

## 使用方式

### 推荐的渐进式实验

```bash
# 实验 1: 基线
python main.py --dataset hicodet ...

# 实验 2: 加 A+D (HOF 改进, 不加额外损失, 最安全)
python main.py --dataset hicodet ... --directional-hof --spatial-refine

# 实验 3: 加 B (先验校准, 对稀有类可能有帮助)
python main.py --dataset hicodet ... --directional-hof --spatial-refine --learnable-prior

# 实验 4: 加 C+E (辅助损失, 进一步提升稀有类)
python main.py --dataset hicodet ... \
    --directional-hof --spatial-refine --learnable-prior \
    --prototype-memory --hierarchical-verb

# 实验 5: 全开
python main.py --dataset hicodet ... \
    --directional-hof --spatial-refine --learnable-prior \
    --prototype-memory --hierarchical-verb --attn-distill
```

### 参数调整建议

| 参数 | 默认值 | 建议范围 | 调大效果 | 调小效果 |
|------|--------|---------|---------|---------|
| `--proto-weight` | 0.1 | 0.05~0.3 | 原型影响更大 | 原型仅做弱正则 |
| `--proto-momentum` | 0.999 | 0.99~0.9999 | 原型更新更慢更稳定 | 原型跟随数据更快 |
| `--hier-weight` | 0.3 | 0.1~0.5 | 粗分类约束更强 | 粗分类仅做辅助 |
| `--attn-distill-weight` | 0.1 | 0.05~0.2 | 注意力被 IoU 约束更紧 | 仅做弱提示 |
| `--debias-weight` | 0.5 | 0.1~1.0 | 去偏更激进 | 去偏更温和 |

---

## 文件修改清单

| 文件 | 改动 | 涉及方向 |
|-----|------|---------|
| `gfin_module/hof.py` | HOFmoduleLayer 加 directional + spatial_refine | A, D |
| `gfin_module/verb_hierarchy.py` | 新文件: 动词粗粒度分组 | E |
| `interaction_head.py` | 加 learnable prior + 返回 hoi_features | B |
| `gfin.py` | ActionPrototypeMemory + 三个新损失 | C, E, F |
| `main.py` | 12 个新 CLI 参数 + 优化器更新 | 全部 |
