# 基于语音自监督表征的低资源 ASR

本项目用于课程大作业方向一：基于语音自监督表征的低资源英文 ASR。项目比较传统 `log-Mel + CTC` 与语音自监督表征 `wav2vec2 / HuBERT + CTC` 在 LibriSpeech 低资源设置下的表现，并进一步分析不同 hidden layer 对识别性能的影响。

当前主结论是：在 5 小时低资源训练、冻结 SSL encoder、只训练 CTC head 的条件下，`wav2vec2` 和 `HuBERT` 的最佳表征层均为 `layer 9`，而不是默认最后层。最终主结果建议使用：

```text
wav2vec2 layer 9 + CTC
HuBERT layer 9 + CTC
```

## 已实现实验

1. `log-Mel + CTC`
   - 传统声学特征 baseline。
   - 输入 waveform，提取 80 维 log-Mel filterbank，经过 CMVN、BiLSTM encoder 和 CTC head。

2. `wav2vec2 + CTC`
   - 使用 `facebook/wav2vec2-base` 提取 SSL 表征。
   - 支持冻结 encoder 和 hidden layer 选择。

3. `HuBERT + CTC`
   - 使用 `facebook/hubert-base-ls960` 提取 SSL 表征。
   - 训练流程、CTC head、字符词表和评价方式与 wav2vec2 保持一致。

4. `wav2vec2 / HuBERT hidden layer ablation`
   - 比较 layer `3, 6, 9, 12`。
   - 结论表明 layer 9 最适合当前低资源 CTC ASR 设置。

当前未实现 k-means discrete units、token 去重、duration modeling、BPE、beam search 和语言模型融合。

## 项目结构

```text
asr_ssl_project/
  README.md
  report.md
  requirements.txt
  configs/
    logmel_ctc.yaml
    wav2vec2_ctc.yaml
    hubert_ctc.yaml
    wav2vec2_ctc_frozen_5h.yaml
    hubert_ctc_frozen_5h.yaml
    wav2vec2_layer_ablation_frozen_5h.yaml
    hubert_layer_ablation_frozen_5h.yaml
  src/
    data/
      librispeech.py
      text.py
      vocab.py
      collate.py
    models/
      logmel_ctc.py
      ssl_ctc.py
    training/
      trainer.py
      metrics.py
      decoding.py
      utils.py
  train.py
  evaluate.py
  run_layer_ablation.py
```

## 环境安装

建议使用独立环境。当前机器上可使用 Anaconda Python：

```powershell
C:\Users\ASUS\Anaconda3\python.exe -m pip install -r requirements.txt
```

如果只缺少 Hugging Face 或音频读取依赖，可单独安装：

```powershell
C:\Users\ASUS\Anaconda3\python.exe -m pip install transformers datasets jiwer soundfile
```

## 数据准备

本项目默认读取本地 LibriSpeech：

```text
asr_ssl_project/
  data/
    LibriSpeech/
      train-clean-100/
      dev-clean/
      test-clean/
```

配置中应保持：

```yaml
data:
  dataset: librispeech
  root: data
  download: false
```

低资源训练通过以下字段控制：

```yaml
max_train_hours: 1.0
```

或：

```yaml
max_train_hours: 5.0
```

如果同时设置 `max_train_hours` 和 `max_train_samples`，代码会先按小时数截断，再按样本数截断。

## 训练命令

进入项目目录：

```powershell
cd C:\Users\ASUS\Desktop\NLP\ASR_work\asr_ssl_project
```

1 小时基础实验：

```powershell
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/logmel_ctc.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/wav2vec2_ctc.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/hubert_ctc.yaml
```

5 小时 frozen SSL 主实验：

```powershell
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/wav2vec2_ctc_frozen_5h.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/hubert_ctc_frozen_5h.yaml
```

5 小时 hidden layer ablation：

```powershell
C:\Users\ASUS\Anaconda3\python.exe run_layer_ablation.py --config configs/wav2vec2_layer_ablation_frozen_5h.yaml
C:\Users\ASUS\Anaconda3\python.exe run_layer_ablation.py --config configs/hubert_layer_ablation_frozen_5h.yaml
```

## 评估命令

```powershell
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/wav2vec2_ctc_frozen_5h.yaml --checkpoint runs/wav2vec2_ctc_frozen_5h/best.pt --split test-clean
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/hubert_ctc_frozen_5h.yaml --checkpoint runs/hubert_ctc_frozen_5h/best.pt --split test-clean
```

ablation 脚本会自动对各层 best checkpoint 运行 test-clean 评估，并写入：

```text
runs/*_layer_ablation_frozen_5h/layer_ablation_results.json
```

## 当前主要结果

5 小时 frozen SSL 主实验：

| System | Setting | Dev WER | Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|
| wav2vec2 + CTC | final layer | 0.9908 | 0.6327 | 0.9897 | 0.6307 |
| HuBERT + CTC | final layer | 0.7142 | 0.2417 | 0.7176 | 0.2448 |

5 小时 layer ablation：

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

## 查看结果文件

每个实验会在 `runs/` 下保存：

- `config.yaml`：训练时使用的配置副本
- `vocab.json`：character-level vocabulary
- `best.pt`：dev WER 最优 checkpoint
- `last.pt`：最后一个 epoch checkpoint
- `metrics.json`：训练过程中的 dev WER/CER
- `dev_predictions.jsonl`：dev 集 reference/hypothesis
- `<split>_metrics.json`：测试集 WER/CER
- `<split>_predictions.jsonl`：测试集 reference/hypothesis
- `train.log`：训练日志

注意：`runs/*/config.yaml` 是训练输出副本。后续修改实验配置时，应编辑 `configs/*.yaml`。

## Future Work

后续可以实现 k-means discrete units，例如 `K=100` 和 `K=500`，比较 continuous SSL representations 与 discrete speech units。

也可以进一步分析 token rate、bitrate、token dedup、duration modeling 和 BPE 对识别性能、压缩率和模型复杂度的影响。这些内容当前尚未实现。

