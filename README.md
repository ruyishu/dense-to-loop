# Dense-to-Loop

把预训练好的 Qwen3-1.7B-Base 转成两次执行同一套权重的 Transformer，再直接用冻结的 Qwen3-8B 做 on-policy distillation。学生转换后的第一步更新来自 OPD。训练中由学生生成回答，老师在学生实际走过的前缀上提供下一 token 的概率。

## 最小 GPU demo

用 Linux、Python 3.11 或 3.12、支持 CUDA 12.8 的 NVIDIA 驱动。模型首次下载约 20 GB。建议至少预留 65 GB 磁盘，供模型缓存、训练快照和临时写入使用。优先用 `3xl20` 或 `1xh20`。GPU 编号按 `CUDA_VISIBLE_DEVICES` 的可见顺序计算。

```bash
git clone https://github.com/ruyishu/dense-to-loop.git
cd dense-to-loop
bash scripts/setup.sh
bash scripts/demo.sh 3xl20
# 单张 96GB H20 可用以下命令
# bash scripts/demo.sh 1xh20 runs/demo-h20
```

Demo 使用真实 8B 老师和 1.7B 学生，取 32 道 GSM8K 训练题做 **4 次更新、累计 512 个回答 tokens**。更新前从留出的题中生成 8 条轨迹，更新前后在这 **同一批前缀** 上计算教师到学生的 KL，分别报告一圈和两圈的值。`runs/demo/opd/metrics.jsonl` 应有四行，`loss` 有限，`updater_grad_norm_sum` 大于零。`rollouts.jsonl` 的 `policy_step` 会随更新递增。最后两行终端输出是固定轨迹上的 KL。

512 tokens 用于验证路径正确和能否更新第二圈。准确率受 48-token 开发生成上限影响。固定轨迹上的 KL 有变化即可检查权重确实生效，KL 的下降幅度和准确率收益要由下面的有对照 pilot 给出。

## 小预算实验

以下以 `3xl20` 为例。先跑每组 8,192 回答 tokens，看耗时、稳定性和截断率。结果可用后，按同一配置恢复到 25 万、100 万和 500 万。设置 `--save-every 20` 可以定期写入可恢复快照。快照包含 FP32 优化器和 master weights，保存时需要额外临时磁盘空间。

```bash
source .venv/bin/activate
d2l prepare-data --output data/pilot --train-count 512 --dev-count 128 --math500
d2l convert --hardware 3xl20 --output runs/pilot/loop-init.pt

for mode in opd sft offline-kd; do
  d2l train --hardware 3xl20 --init runs/pilot/loop-init.pt \
    --data data/pilot/train.jsonl --mode "$mode" --output "runs/pilot/loop-$mode" \
    --budget 8192 --max-new 256 --save-every 20
done

for mode in opd sft; do
  d2l train --hardware 3xl20 --dense \
    --data data/pilot/train.jsonl --mode "$mode" --output "runs/pilot/dense-$mode" \
    --budget 8192 --max-new 256 --save-every 20
done

for mode in opd sft offline-kd; do
  d2l eval --hardware 3xl20 --checkpoint "runs/pilot/loop-$mode/last.pt" \
    --data data/pilot/dev.jsonl --max-new 512 --output "reports/loop-$mode.jsonl"
done
d2l report --left reports/loop-sft.jsonl --right reports/loop-opd.jsonl

# 同一 OPD checkpoint 只读出第一圈，与两圈结果按题比较
d2l eval --hardware 3xl20 --checkpoint runs/pilot/loop-opd/last.pt \
  --data data/pilot/dev.jsonl --depth 1 --max-new 512 --output reports/loop-opd-one.jsonl
d2l report --left reports/loop-opd-one.jsonl --right reports/loop-opd.jsonl
```

恢复时使用同一 `--output` 和同一训练配置，通过 `--resume --budget 250000` 指定新的累计目标。若有独立的教师参考回答，可在 JSONL 中给每行加上由同一 8B 和同一 chat template 生成的 `response` 字段，供 SFT 和 offline-KD 共用。默认在线生成教师参考回答并计入对应组成本。OPD 只用 `prompt`。同一初始化、prompt 清单、训练预算下先确认 Loop+OPD 超过 Loop+SFT，再加入离线 KL 和 dense 对照。

### 评测和判断

| 指标 | 用途 |
|---|---|
| 开发集准确率、截断率、生成长度 | 判断训练方向和输出上限是否合适 |
| 固定前缀的一圈／两圈 KL，外加相同题目的正确率变化 | 判断第二圈是否学到额外信息，答案评分计算错变对和对变错 |
| MATH-500、OlympiadBench、AIME24–26 | 正式结果。MATH-500 有内置下载入口。其他基准通过相同格式的 `id`、`prompt`、`answer` JSONL 输入 |
| 学生 rollout、教师生成与打分、学生回放 token 数和耗时 | 计算匹配和吞吐。按同一累计回答 token 预算比较，同时报告分阶段成本 |

GSM8K 的训练、开发和测试清单由固定 seed 生成，JSONL 文件 SHA-256 写入训练快照。正式主表应锁定原始题集版本，核查和训练题的重叠。要用 TaH2 的数学子集时，将上游 prompts 与教师答案整理成同一 JSONL 格式。原论文的 1.7B 固定两圈完整 SFT 参考为 AIME 24–26 均分 14.6、MATH-500 76.1、OlympiadBench 42.1。原文采样和评分协议与默认 pilot 不同，对比时用相同的题集、解码预算和评分器。[TaH2 论文](https://arxiv.org/abs/2609.35748) · [TaH2 数据](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool)

## 实现细节

- 回放以原始 token ID 为准。回答索引从 prompt 最后一个预测位置开始。词表使用两个 tokenizer 共用的 token ID 集，剔除 LM head 中未分配给 tokenizer 的补齐行。蒸馏损失是逐回答位置的全有效词表 forward KL。
- 固定循环复用 Transformers 的 Qwen3 权重和层组件。两圈预测概率做等权平均。第二圈只看第一圈的因果上下文及自身第二圈 KV。updater 采用 TaH2 附录中的融合、RMSNorm 和 SwiGLU 结构，新增参数由固定种子初始化。[TaH2 固定两圈定义](https://arxiv.org/html/2609.35748v1) · [官方实现](https://github.com/thu-nics/TaH)
- 这是普通 PyTorch SDPA 参考实现。同一代码负责逐 token KV 缓存采样和训练回放，训练支持梯度检查点和全词表投影分块。训练按序列逐条累积，优先保证一致性，长输出和百万级 rollout 的实际吞吐需先测。它使用层切分、阶段顺序执行，未接入上游 Triton / mini-SGLang 的动态批处理。
- `3xl20` 把学生 decoder 切到前两张卡，教师驻第三张。`1xh20` 在同一张卡上顺序采样、打分和更新，并保留两份模型权重。`2xl20` 将两个模型各放一张卡，适合短 demo 或 `--updater-only`，全参训练峰值需要实测。`--updater-only` 适合诊断，正式结构对照默认全参更新。
- 版本由 HF revision、训练 JSONL 哈希和配置记录约束。要复用相同远端模型，可从转换日志取 revision 传给 dense 组。`--student` 与 `--teacher` 也支持本地 Qwen3 模型目录。`--resume` 恢复优化器、训练 token、prompt 游标和随机状态。每个输出目录用单个训练进程写入。

```bash
python -m unittest discover -s tests -v
```

CPU 测试使用本地生成的小型 Qwen3，涵盖第一圈和 dense 对齐、KV 缓存与完整计算一致、因果性、分块 KL 和 updater 梯度、转架构后直接 OPD、SFT、离线 KL 及恢复。真实 Qwen3 权重在 L20/H20 上的显存、速度和精度由 GPU demo 验证。
