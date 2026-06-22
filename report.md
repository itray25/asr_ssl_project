# 低资源 ASR 实验简短报告

## 1. 任务目标

本项目研究语音自监督表征在低资源英文 ASR 中的作用。系统输入一段语音 waveform，输出对应英文文本转写。实验比较传统 `log-Mel + CTC` 与 `wav2vec2 / HuBERT + CTC`，并进一步分析不同 hidden layer 和不同训练数据规模对识别性能的影响。

实验使用 LibriSpeech `train-clean-100` 的低资源子集进行训练，开发集和测试集分别为 `dev-clean` 与 `test-clean`。评价指标为 WER 和 CER。

## 2. 方法

### 2.1 log-Mel + CTC Baseline

传统 baseline 使用 80 维 log-Mel filterbank，经过 CMVN 后输入 BiLSTM encoder，最后通过线性层和 CTC loss 学习字符级 ASR。

```text
waveform -> log-Mel -> CMVN -> BiLSTM -> CTC head -> text
```

### 2.2 SSL + CTC

SSL 系统使用 Hugging Face 预训练模型提取帧级 hidden states，再接线性 CTC head。实验中冻结 SSL encoder，只训练 CTC head，以突出预训练表征本身的作用。

```text
waveform -> wav2vec2 / HuBERT -> hidden states -> CTC head -> text
```

### 2.3 Hidden Layer Ablation

为了分析不同层表征对 ASR 的影响，分别选择 layer `3, 6, 9, 12` 作为 CTC head 的输入。除 hidden layer 外，其余训练数据、训练轮数、解码方式均保持一致。

### 2.4 Training-Scale Study

在 ablation 发现最佳层为 layer 9 后，进一步固定 `hidden_layer=9`，比较 `1h / 5h / 10h` 标注数据规模下 wav2vec2 和 HuBERT 的表现。

## 3. 实验设置

文本规范化包括：

- 转小写
- 去除无关标点
- 保留 `a-z`、空格和 apostrophe

输出单位为 character-level vocabulary。训练目标为 CTC loss，解码方式为 greedy CTC decoding。

主要训练设置：

| Experiment | Train Hours | Encoder | Epochs | Decoding |
|---|---:|---|---:|---|
| log-Mel + CTC | 1h | BiLSTM | 10 | greedy |
| wav2vec2 + CTC | 1h / 5h / 10h | frozen wav2vec2-base | 5 / 10 | greedy |
| HuBERT + CTC | 1h / 5h / 10h | frozen HuBERT-base | 5 / 10 | greedy |
| Layer ablation | 5h | frozen SSL encoder | 10 | greedy |

## 4. 结果

### 4.1 1 小时低资源实验

| System | Dev WER | Dev CER | Observation |
|---|---:|---:|---|
| log-Mel + CTC | 1.0000 | 0.9977 | 几乎无法学习有效转写 |
| wav2vec2 + CTC final layer | 0.9969 | 0.7396 | 能输出字符片段，但词级识别很弱 |
| HuBERT + CTC final layer | 0.8384 | 0.3322 | 明显优于 log-Mel 和 wav2vec2 |

1 小时设置下，log-Mel baseline 基本输出空串或极少字符。wav2vec2 final layer 能降低 CER，但 WER 仍接近 1。HuBERT final layer 能生成较完整的文本片段，说明其表征更适合低资源字符级 CTC 建模。

### 4.2 5 小时 Frozen SSL 主实验

| System | Setting | Best Dev WER | Best Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|
| wav2vec2 + CTC | final layer | 0.9908 | 0.6327 | 0.9897 | 0.6307 |
| HuBERT + CTC | final layer | 0.7142 | 0.2417 | 0.7176 | 0.2448 |

在默认最后层设置下，HuBERT 显著优于 wav2vec2。HuBERT 的 test WER/CER 为 `0.7176 / 0.2448`，而 wav2vec2 的 test WER/CER 为 `0.9897 / 0.6307`。

### 4.3 Hidden Layer Ablation

| Model | Layer | Best Dev WER | Best Dev CER | Test WER | Test CER |
|---|---:|---:|---:|---:|---:|
| wav2vec2 | 3 | 0.9343 | 0.4574 | 0.9344 | 0.4602 |
| wav2vec2 | 6 | 0.7014 | 0.2575 | 0.7043 | 0.2610 |
| wav2vec2 | 9 | **0.6166** | **0.1988** | **0.6319** | **0.2076** |
| wav2vec2 | 12 | 0.9910 | 0.6352 | 0.9901 | 0.6296 |
| HuBERT | 3 | 0.9624 | 0.5157 | 0.9611 | 0.5146 |
| HuBERT | 6 | 0.7583 | 0.2734 | 0.7598 | 0.2777 |
| HuBERT | 9 | **0.6019** | **0.1962** | **0.6036** | **0.1986** |
| HuBERT | 12 | 0.7148 | 0.2420 | 0.7178 | 0.2452 |

两种 SSL 模型的最佳层均为 layer 9。该结果说明，最后层并不一定最适合低资源 ASR。中高层表征可能在声学细节和音素级抽象之间取得更好平衡。

### 4.4 训练数据规模实验

| Model | Train Hours | Dev WER | Dev CER | Test WER | Test CER |
|---|---:|---:|---:|---:|---:|
| wav2vec2 layer 9 | 1h | 0.7732 | 0.3430 | 0.7935 | 0.3592 |
| wav2vec2 layer 9 | 5h | 0.6166 | 0.1988 | 0.6319 | 0.2076 |
| wav2vec2 layer 9 | 10h | **0.6008** | **0.1923** | **0.6112** | **0.1994** |
| HuBERT layer 9 | 1h | 0.7037 | 0.2639 | 0.7075 | 0.2672 |
| HuBERT layer 9 | 5h | 0.6019 | 0.1962 | 0.6036 | 0.1986 |
| HuBERT layer 9 | 10h | **0.5975** | **0.1932** | **0.6032** | **0.1967** |

从 1h 增加到 5h 时，两种模型均获得显著提升。wav2vec2 的 test WER 从 `0.7935` 降至 `0.6319`，HuBERT 的 test WER 从 `0.7075` 降至 `0.6036`。从 5h 增加到 10h 后收益明显变小，尤其 HuBERT 的 test WER 基本保持在 `0.603` 附近，说明 frozen encoder + linear CTC head + greedy decoding 已接近当前设置下的瓶颈。

### 4.5 K-means 离散语音单元分析

为了进一步比较 continuous SSL representations 与 discrete speech units，本实验使用最佳 ASR 设置中的 layer 9 表征进行 K-means 离散化。具体做法是从 `dev-clean` 提取 HuBERT / wav2vec2 layer 9 的帧级 hidden states，并分别训练 `K=100` 和 `K=500` 的 K-means codebook。随后将每帧连续向量映射为离散 unit id，并统计 token rate、bitrate、去重后的 token rate、压缩率和 codebook 使用情况。

| Model | Layer | K | Token rate | Dedup token rate | Bitrate | Dedup bitrate | Compression | Used units | Entropy | Effective units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuBERT | 9 | 100 | 49.91 | 26.20 | 331.59 | 174.07 | 1.90x | 100/100 | 6.53 | 92.71 |
| HuBERT | 9 | 500 | 49.91 | 29.13 | 447.48 | 261.20 | 1.71x | 500/500 | 8.75 | 430.31 |
| wav2vec2 | 9 | 100 | 49.91 | 30.08 | 331.59 | 199.88 | 1.66x | 100/100 | 6.46 | 88.09 |
| wav2vec2 | 9 | 500 | 49.91 | 35.22 | 447.48 | 315.81 | 1.42x | 500/500 | 8.73 | 423.42 |

| Model | K | Avg run length | Max run length |
|---|---:|---:|---:|
| HuBERT | 100 | 1.90 frames | 33 frames |
| HuBERT | 500 | 1.71 frames | 36 frames |
| wav2vec2 | 100 | 1.66 frames | 21 frames |
| wav2vec2 | 500 | 1.42 frames | 17 frames |

原始 token rate 均约为 `49.91 tokens/sec`，这是由 SSL encoder 的帧率决定的。`K=500` 相比 `K=100` 使用更大的 codebook，因此 bitrate 更高，但 effective units 也明显更多，表示离散单元能刻画更细粒度的声学差异。对连续重复 unit 进行 dedup 后，token rate 和 bitrate 均显著下降，说明离散 unit 序列存在较强的时间冗余。

HuBERT 的 dedup 压缩率高于 wav2vec2，且平均 run length 更长，说明 HuBERT layer 9 的离散单元在时间上更加稳定；wav2vec2 的 dedup token rate 更高，说明其 unit 序列变化更频繁。两种模型的 codebook 都被充分使用，表明 K-means 离散化没有出现明显的 codebook collapse。

## 5. 分析

### 5.1 SSL 表征优于传统 log-Mel baseline

log-Mel baseline 在 1 小时训练数据下几乎无法收敛到可用 ASR 系统，说明从零训练声学模型对标注数据量要求较高。相比之下，SSL 表征即使冻结 encoder，也能提供更强的语音内容信息。

### 5.2 hidden layer 选择非常关键

ablation 显示，wav2vec2 和 HuBERT 的 layer 9 均取得最佳结果。特别是 wav2vec2，默认最后层表现很弱，但 layer 9 将 test WER 从约 `0.9901` 降至 `0.6319`。因此，wav2vec2 效果差并不是模型本身不可用，而是最后层表征不适合当前 frozen CTC 任务。

### 5.3 数据规模收益存在递减

固定最佳 layer 9 后，1h 到 5h 的性能提升最明显；5h 到 10h 的提升较小。HuBERT 在 1h 条件下明显优于 wav2vec2，但在 10h 时两者差距缩小，说明随着标注数据增加，wav2vec2 layer 9 也能逐渐被 CTC head 更好利用。

### 5.4 离散单元的压缩性与稳定性

K-means 离散化将 768 维连续 SSL hidden states 转换为低 bitrate 的符号序列。相比原始帧级 unit 序列，dedup 能显著降低 token rate 和 bitrate，例如 HuBERT `K=100` 从 `331.59 bps` 降至 `174.07 bps`。但 dedup 同时移除了显式时长信息，因此如果后续将离散 unit 用于 unit-based ASR 或 TTS，需要额外的 duration modeling 来恢复时间结构。

在相同 codebook size 下，HuBERT 的 dedup compression ratio 高于 wav2vec2，说明其离散 unit 在相邻帧之间更稳定。这与 HuBERT 在连续表征 ASR 中略优于 wav2vec2 的结果一致，表明更稳定的中高层语音表征可能更适合低资源识别和离散化建模。

### 5.5 错误类型

最佳模型仍存在明显错误，主要包括：

- 词边界错误，例如多个词粘连
- 拼写近似错误，例如省略元音或混淆辅音
- 音近替换，例如 `night` / `nigt`
- 长句中的删除和重复

这些错误与 greedy CTC decoding、无语言模型和低资源训练条件有关。

## 6. 结论

本实验表明，语音自监督表征能显著改善低资源 ASR。HuBERT 和 wav2vec2 的中高层表征均优于传统 log-Mel baseline，其中 HuBERT layer 9 在 10h 设置下取得当前最佳 test-clean 结果：

```text
WER = 0.6032
CER = 0.1967
```

因此，最终系统建议采用 `HuBERT layer 9 + CTC` 作为当前阶段的最佳低资源 ASR 系统，同时将 `wav2vec2 layer 9 + CTC` 作为主要 SSL 对比系统。进一步的 K-means 实验表明，最佳层的连续 SSL 表征可以被压缩为较低 bitrate 的离散语音单元；其中 HuBERT 离散单元在时间上更稳定，dedup 后具有更高压缩率。

## 7. 后续工作

后续可以继续扩展：

- fine-tuning SSL encoder，并为 encoder/head 设置不同学习率
- beam search 或语言模型融合
- 使用离散语音单元重新训练 unit-based ASR，并与 continuous CTC 系统直接比较 WER/CER
- 对离散 unit 序列进一步使用 BPE 或语言模型建模
- 引入 duration modeling，分析 dedup unit 序列在保留语音内容和时长信息之间的权衡

