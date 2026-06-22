# 基于语音自监督表征的低资源 ASR

本项目用于课程大作业方向一：基于语音自监督表征的低资源英文 ASR。项目比较传统 `log-Mel + CTC` 与语音自监督表征 `wav2vec2 / HuBERT + CTC` 在 LibriSpeech 低资源设置下的表现，并进一步分析 hidden layer 选择和训练数据规模对识别性能的影响。

当前主结论：

- 传统 `log-Mel + CTC` 在 1 小时低资源设置下几乎无法得到可用转写。
- `wav2vec2` 和 `HuBERT` 的最佳表征层均为 `layer 9`，而不是默认最后层。
- 在最佳 layer 9 设置下，训练数据从 1h 增加到 5h 带来显著提升；从 5h 增加到 10h 后收益变小。
- 当前最佳结果来自 `HuBERT layer 9 + CTC, 10h`，test-clean `WER=0.6032`，`CER=0.1967`。
- 已基于最佳 layer 9 表征实现 K-means discrete units 分析，比较 `K=100/500` 下的 token rate、bitrate、dedup 压缩率和 codebook 使用情况。

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

5. `layer 9 training-scale study`
   - 固定使用上一步实验所得的最佳layer： layer 9，比较 `1h / 5h / 10h` 训练数据规模。
   - 用于分析低资源条件下数据规模对性能的影响。

6. `K-means discrete speech units`
   - 基于最佳 layer 9 continuous SSL representations 提取帧级 hidden states。
   - 使用 K-means 构建 `K=100/500` 离散 codebook。
   - 分析 token rate、bitrate、token dedup、duration/run-length、codebook usage 和 entropy。

当前未实现 unit-based ASR 训练、BPE、beam search 和语言模型融合。

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
    wav2vec2_layer9_frozen_1h.yaml
    wav2vec2_layer9_frozen_5h.yaml
    wav2vec2_layer9_frozen_10h.yaml
    hubert_layer9_frozen_1h.yaml
    hubert_layer9_frozen_5h.yaml
    hubert_layer9_frozen_10h.yaml
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
  analyze_kmeans_units.py
```

## 环境安装

建议使用独立环境。当前项目可使用本地 `.conda` 环境：

```powershell
& ".conda\python.exe" -m pip install -r requirements.txt
```

如果使用其他 Anaconda/Miniconda 环境，也可以将上面的 Python 路径替换为对应环境的 `python.exe`。

## 数据准备

本项目默认读取本地 LibriSpeech。标准目录为：

```text
asr_ssl_project/
  data/
    LibriSpeech/
      train-clean-100/
      dev-clean/
      test-clean/
```

如果解压后形成如下多一层目录的结构，当前 `src/data/librispeech.py` 也可以读取：

```text
asr_ssl_project/
  data/
    LibriSpeech/
      train-clean-100/LibriSpeech/train-clean-100/
      dev-clean/LibriSpeech/dev-clean/
      test-clean/LibriSpeech/test-clean/
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
max_train_hours: 5.0
max_train_hours: 10.0
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

最佳 layer 9 的训练数据规模实验：

```powershell
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/wav2vec2_layer9_frozen_1h.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/wav2vec2_layer9_frozen_5h.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/wav2vec2_layer9_frozen_10h.yaml

C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/hubert_layer9_frozen_1h.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/hubert_layer9_frozen_5h.yaml
C:\Users\ASUS\Anaconda3\python.exe train.py --config configs/hubert_layer9_frozen_10h.yaml
```

说明：5h 的 layer 9 结果也可以直接复用 ablation 中的 `layer_9` 目录。当前项目已将这两份 5h layer 9 结果复制到统一目录：

```text
runs/wav2vec2_layer9_frozen_5h/
runs/hubert_layer9_frozen_5h/
```

## 评估命令

5 小时主实验：

```powershell
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/wav2vec2_ctc_frozen_5h.yaml --checkpoint runs/wav2vec2_ctc_frozen_5h/best.pt --split test-clean
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/hubert_ctc_frozen_5h.yaml --checkpoint runs/hubert_ctc_frozen_5h/best.pt --split test-clean
```

layer 9 数据规模实验：

```powershell
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/wav2vec2_layer9_frozen_1h.yaml --checkpoint runs/wav2vec2_layer9_frozen_1h/best.pt --split test-clean
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/wav2vec2_layer9_frozen_5h.yaml --checkpoint runs/wav2vec2_layer9_frozen_5h/best.pt --split test-clean
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/wav2vec2_layer9_frozen_10h.yaml --checkpoint runs/wav2vec2_layer9_frozen_10h/best.pt --split test-clean

C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/hubert_layer9_frozen_1h.yaml --checkpoint runs/hubert_layer9_frozen_1h/best.pt --split test-clean
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/hubert_layer9_frozen_5h.yaml --checkpoint runs/hubert_layer9_frozen_5h/best.pt --split test-clean
C:\Users\ASUS\Anaconda3\python.exe evaluate.py --config configs/hubert_layer9_frozen_10h.yaml --checkpoint runs/hubert_layer9_frozen_10h/best.pt --split test-clean
```

ablation 脚本会自动对各层 best checkpoint 运行 test-clean 评估，并写入：

```text
runs/*_layer_ablation_frozen_5h/layer_ablation_results.json
```

## K-means 离散单元分析

`analyze_kmeans_units.py` 用于将 SSL continuous hidden states 离散化为 K-means speech units，并统计 token rate、bitrate、去重压缩率、duration/run-length 和 codebook 使用情况。该脚本不重新训练 ASR 模型，主要用于分析 continuous representations 与 discrete speech units 的压缩性和复杂度。

先运行一个小规模 smoke test：

```powershell
& ".conda\python.exe" analyze_kmeans_units.py --config configs/hubert_layer9_frozen_10h.yaml --split dev-clean --k 20 --max-frames 2000 --max-utts 3 --output runs/kmeans_units/smoke_hubert_layer9_k20.json
```

正式实验命令：

```powershell
& ".conda\python.exe" analyze_kmeans_units.py --config configs/hubert_layer9_frozen_10h.yaml --split dev-clean --k 100 --max-frames 200000 --output runs/kmeans_units/hubert_layer9_k100_dev.json
& ".conda\python.exe" analyze_kmeans_units.py --config configs/hubert_layer9_frozen_10h.yaml --split dev-clean --k 500 --max-frames 200000 --output runs/kmeans_units/hubert_layer9_k500_dev.json

& ".conda\python.exe" analyze_kmeans_units.py --config configs/wav2vec2_layer9_frozen_10h.yaml --split dev-clean --k 100 --max-frames 200000 --output runs/kmeans_units/wav2vec2_layer9_k100_dev.json
& ".conda\python.exe" analyze_kmeans_units.py --config configs/wav2vec2_layer9_frozen_10h.yaml --split dev-clean --k 500 --max-frames 200000 --output runs/kmeans_units/wav2vec2_layer9_k500_dev.json
```

输出文件位于：

```text
runs/kmeans_units/*.json
```

主要字段包括：

- `token_rate`：原始离散 unit 每秒 token 数
- `dedup_token_rate`：合并连续重复 unit 后的每秒 token 数
- `bitrate` / `dedup_bitrate`：离散表示的信息率
- `dedup_compression_ratio`：去重压缩率
- `used_units`：实际使用的 codebook 数量
- `entropy_bits` / `effective_num_units`：unit 分布熵和有效单元数
- `avg_run_length_frames` / `max_run_length_frames`：连续重复 unit 的时长统计

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

最佳 layer 9 的训练数据规模实验：

| Model | Train Hours | Dev WER | Dev CER | Test WER | Test CER |
|---|---:|---:|---:|---:|---:|
| wav2vec2 layer 9 | 1h | 0.7732 | 0.3430 | 0.7935 | 0.3592 |
| wav2vec2 layer 9 | 5h | 0.6166 | 0.1988 | 0.6319 | 0.2076 |
| wav2vec2 layer 9 | 10h | **0.6008** | **0.1923** | **0.6112** | **0.1994** |
| HuBERT layer 9 | 1h | 0.7037 | 0.2639 | 0.7075 | 0.2672 |
| HuBERT layer 9 | 5h | 0.6019 | 0.1962 | 0.6036 | 0.1986 |
| HuBERT layer 9 | 10h | **0.5975** | **0.1932** | **0.6032** | **0.1967** |

K-means 离散单元分析，基于 `dev-clean` 和 layer 9 hidden states：

| Model | K | Token rate | Dedup token rate | Bitrate | Dedup bitrate | Compression | Used units | Entropy | Effective units |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuBERT | 100 | 49.91 | 26.20 | 331.59 | 174.07 | 1.90x | 100/100 | 6.53 | 92.71 |
| HuBERT | 500 | 49.91 | 29.13 | 447.48 | 261.20 | 1.71x | 500/500 | 8.75 | 430.31 |
| wav2vec2 | 100 | 49.91 | 30.08 | 331.59 | 199.88 | 1.66x | 100/100 | 6.46 | 88.09 |
| wav2vec2 | 500 | 49.91 | 35.22 | 447.48 | 315.81 | 1.42x | 500/500 | 8.73 | 423.42 |

| Model | K | Avg run length | Max run length |
|---|---:|---:|---:|
| HuBERT | 100 | 1.90 frames | 33 frames |
| HuBERT | 500 | 1.71 frames | 36 frames |
| wav2vec2 | 100 | 1.66 frames | 21 frames |
| wav2vec2 | 500 | 1.42 frames | 17 frames |

HuBERT 的 dedup compression ratio 高于 wav2vec2，说明其离散 unit 序列在时间上更稳定；`K=500` 的 effective units 更多，但 bitrate 和 dedup 后复杂度也更高。

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

K-means 离散单元分析会在 `runs/kmeans_units/` 下保存 JSON 统计文件，例如：

- `hubert_layer9_k100_dev.json`
- `hubert_layer9_k500_dev.json`
- `wav2vec2_layer9_k100_dev.json`
- `wav2vec2_layer9_k500_dev.json`

注意：`runs/*/config.yaml` 是训练输出副本。后续修改实验配置时，应编辑 `configs/*.yaml`。

## Future Work

后续可以继续扩展：

- 使用离散语音单元重新训练 unit-based ASR，并与 continuous CTC 系统直接比较 WER/CER。
- 对离散 unit 序列应用 BPE，分析更长 unit pattern 对序列长度和识别建模的影响。
- 引入 duration modeling，补偿 token dedup 后丢失的显式时长信息。
- 加入 beam search 或语言模型融合，降低 greedy CTC decoding 带来的词边界和拼写错误。
- fine-tune SSL encoder，并为 encoder 和 CTC head 设置不同学习率。

