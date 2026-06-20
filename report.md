# 低资源 ASR 实验简短报告

## 1. 任务目标

本项目研究语音自监督表征在低资源英文 ASR 中的作用。系统输入一段语音 waveform，输出对应英文文本转写。实验比较传统 `log-Mel + CTC` 与 `wav2vec2 / HuBERT + CTC`，并进一步分析不同 hidden layer 的表征质量。

实验使用 LibriSpeech `train-clean-100` 的低资源子集进行训练，开发集和测试集分别为 `dev-clean` 与 `test-clean`。评价指标为 WER 和 CER。

## 2. 方法

### 2.1 log-Mel + CTC Baseline

传统 baseline 使用 80 维 log-Mel filterbank，经过 CMVN 后输入 BiLSTM encoder，最后通过线性层和 CTC loss 学习字符级 ASR。

```text
waveform -> log-Mel -> CMVN -> BiLSTM -> CTC head -> text
```

### 2.2 SSL + CTC

SSL 系统使用 Hugging Face 预训练模型提取帧级 hidden states，再接线性 CTC head。实验中默认冻结 SSL encoder，只训练 CTC head，以突出预训练表征本身的作用。

```text
waveform -> wav2vec2 / HuBERT -> hidden states -> CTC head -> text
```

### 2.3 Hidden Layer Ablation

为了分析不同层表征对 ASR 的影响，分别选择 layer `3, 6, 9, 12` 作为 CTC head 的输入。除 hidden layer 外，其余训练数据、训练轮数、解码方式均保持一致。

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
| wav2vec2 + CTC | 1h / 5h | frozen wav2vec2-base | 5 / 10 | greedy |
| HuBERT + CTC | 1h / 5h | frozen HuBERT-base | 5 / 10 | greedy |
| Layer ablation | 5h | frozen SSL encoder | 10 | greedy |

## 4. 结果

### 4.1 1 小时低资源实验

| System | Dev WER | Dev CER | Observation |
|---|---:|---:|---|
| log-Mel + CTC | 1.0000 | 0.9977 | 几乎无法学习有效转写 |
| wav2vec2 + CTC | 0.9969 | 0.7396 | 能输出字符片段，但词级识别很弱 |
| HuBERT + CTC | 0.8384 | 0.3322 | 明显优于 log-Mel 和 wav2vec2 |

1 小时设置下，log-Mel baseline 基本输出空串或极少字符。wav2vec2 能降低 CER，但 WER 仍接近 1。HuBERT 能生成较完整的文本片段，说明其表征更适合低资源字符级 CTC 建模。

### 4.2 5 小时 Frozen SSL 主实验

| System | Setting | Best Dev WER | Best Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|
| wav2vec2 + CTC | final layer | 0.9908 | 0.6327 | 0.9897 | 0.6307 |
| HuBERT + CTC | final layer | 0.7142 | 0.2417 | 0.7176 | 0.2448 |

在默认最后层设置下，HuBERT 仍明显优于 wav2vec2。HuBERT 的 test WER/CER 为 `0.7176 / 0.2448`，而 wav2vec2 的 test WER/CER 为 `0.9897 / 0.6307`。

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

## 5. 分析

### 5.1 SSL 表征优于传统 log-Mel baseline

log-Mel baseline 在 1 小时训练数据下几乎无法收敛到可用 ASR 系统，说明从零训练声学模型对标注数据量要求较高。相比之下，SSL 表征即使冻结 encoder，也能提供更强的语音内容信息。

### 5.2 HuBERT 在默认最后层下优于 wav2vec2

使用默认最后层时，HuBERT 显著优于 wav2vec2。这可能与 HuBERT 的 masked prediction of discrete units 预训练目标有关，其高层表征更直接编码了语音内容。

### 5.3 layer 9 是最佳 ASR 表征层

ablation 显示，wav2vec2 和 HuBERT 的 layer 9 均取得最佳结果。特别是 wav2vec2，默认最后层表现很弱，但 layer 9 将 test WER 从约 `0.9901` 降至 `0.6319`。因此，wav2vec2 效果差并不是模型本身不可用，而是最后层表征不适合当前 frozen CTC 任务。

### 5.4 错误类型

最佳模型仍存在明显错误，主要包括：

- 词边界错误，例如多个词粘连
- 拼写近似错误，例如省略元音或混淆辅音
- 音近替换，例如 `night` / `nigt`
- 长句中的删除和重复

这些错误与 greedy CTC decoding、无语言模型和低资源训练条件有关。

## 6. 结论

本实验表明，语音自监督表征能显著改善低资源 ASR。HuBERT 和 wav2vec2 的中高层表征均优于传统 log-Mel baseline，其中 HuBERT layer 9 取得最佳 test-clean 结果：

```text
WER = 0.6036
CER = 0.1986
```

因此，最终系统建议采用 `HuBERT layer 9 + CTC` 作为当前阶段的最佳低资源 ASR 系统，同时将 `wav2vec2 layer 9 + CTC` 作为主要 SSL 对比系统。

## 7. 后续工作

后续可以继续扩展：

- fine-tuning SSL encoder，并为 encoder/head 设置不同学习率
- beam search 或语言模型融合
- k-means discrete units，例如 `K=100/500`
- continuous vs discrete units 对比
- token rate、bitrate 和压缩效率分析

