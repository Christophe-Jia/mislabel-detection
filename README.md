# Learning from Training Dynamics: Identifying Mislabeled Data Beyond Manually Designed Features

## Abstract
---
>While mislabeled or ambiguously-labeled samples in the training set could negatively affect the performance of deep models, diagnosing the dataset and identifying mislabeled samples helps to improve the generalization power. *Training dynamics*, i.e., the traces left by iterations of optimization algorithms, have recently been proven to be effective to localize mislabeled samples with hand-crafted features.
In this paper, beyond manually designed features, we introduce a novel learning-based solution, leveraging a *noise detector*, instanced by an LSTM network, which learns to predict whether a sample was mislabeled using the raw training dynamics as input.
Specifically, the proposed method trains the noise detector in a supervised manner using the dataset with synthesized label noises and can adapt to various datasets (either naturally or synthesized label-noised) without retraining.
We conduct extensive experiments to evaluate the proposed method.
We train the noise detector based on the synthesized label-noised CIFAR dataset and test such noise detectors on Tiny ImageNet, CUB-200, Caltech-256, WebVision, and Clothing1M.
Results show that the proposed method precisely detects mislabeled samples on various datasets without further adaptation, and outperforms state-of-the-art methods.
Besides, more experiments demonstrate that mislabel identification can guide a label correction, namely data debugging, providing orthogonal improvements of algorithm-centric state-of-the-art techniques from the data aspect.

## Requirements
---
- pytorch >= 1.7.1
- torchvision >= 0.4
- scikit-learn
- numpy
- pandas

## Project Structure
---
```
mislabel-detection/
├── runner.py                  # Core pipeline: dataset creation, training, evaluation, training dynamics generation
├── train_detector.py          # Train LSTM noise detector on training dynamics
├── losses.py                  # Loss functions (cross-entropy, Reed soft/hard)
├── util.py                    # Utilities: AverageMeter, Welford statistics, training dynamics I/O
├── models/                    # Classification model architectures
│   ├── resnet.py              #   ResNet (various depths)
│   ├── wide_resnet.py         #   WideResNet
│   ├── densenet.py            #   DenseNet
│   ├── PreResNet.py           #   Pre-activation ResNet
│   ├── vgg.py                 #   VGG
│   ├── conv4.py               #   Conv4
│   └── lenet.py               #   LeNet / LeNetMNIST
├── detector_model/            # Noise detector
│   ├── lstm.py                #   LSTM binary classifier
│   └── predict.py             #   Ranking & evaluation (get_order)
├── data_debug_dm/             # Data debugging experiments (Chapter 4.3)
│   ├── cub_200_201/           #   CUB-200 label correction
│   └── mini_webvision/        #   WebVision label correction
├── generate_td.sh             # Step 1: Generate training dynamics
├── small_dataset_sym_denoise.sh  # Step 3a: Denoise small datasets (CIFAR, Tiny ImageNet)
└── large_dataset_denoise.sh   # Step 3b: Denoise large datasets (WebVision, Clothing1M)
```

### Pipeline Overview

The project follows a 3-phase pipeline:

1. **Generate Training Dynamics** (`generate_td.sh` -> `runner.py`): Train a classification model and record per-sample prediction probabilities across all epochs.
2. **Train Noise Detector** (`train_detector.py`): Train an LSTM to classify samples as clean or mislabeled, using the training dynamics as input sequences.
3. **Denoise & Retrain** (`small_dataset_sym_denoise.sh` / `large_dataset_denoise.sh` -> `runner.py`): Use the trained detector to rank and remove suspected mislabeled samples, then retrain on the cleaned data.

## Paper Replication in Chapters 4.1 and 4.2
---
### Datasets
We run experiments on 5 **small** datasets:
- CIFAR-10
- CIFAR-100
- Tiny ImageNet
- CUB-200-2011
- Caltech-256

... and 2 **large** datasets:
- WebVision50 (subset of WebVision)
- Clothing100K (subset of Clothing1M)

We use the same subset as [AUM](https://github.com/asappresearch/aum/tree/master/examples/paper_replication) for these two subsets. Click [Here](https://drive.google.com/file/d/1rr2nvnnBMsbo1qcU3i3urJsDw86PJ9tR/view?usp=sharing) to download and untar the file to access CUB-200-2011 and Caltech-256.

### Replication Steps

1. Acquire metadata and training dynamics for manually corrupted or real-world datasets.
2. Train an LSTM model as a noise detector.
3. Retrain a new model on cleaned data:
   - Metrics of label noise detection on synthesized datasets (CIFAR-10/100, Tiny ImageNet) and retraining on clean data
   - Less overfitting towards noisy labels on real-world datasets (WebVision50 and Clothing100K)

### STEP 1: Acquire Metadata and Training Dynamics

```sh
generate_td.sh <datadir> <dataset> <seed> <noise_ratio> <noise_type> <net_type> <depth>

# Generate td for small datasets [no manual corruption]
CUDA_VISIBLE_DEVICES=0 ./generate_td.sh "/root/codespace/datasets" "cifar10" 1 0. "uniform" "resnet" 32
# Generate td for small datasets [uniform 0.2 noisy]
CUDA_VISIBLE_DEVICES=0 ./generate_td.sh "/root/codespace/datasets" "tiny_imagenet" 1 0.2 "uniform" "resnet" 32
# Generate td for large datasets [noise_ratio and noise_type are mute]
CUDA_VISIBLE_DEVICES=0 ./generate_td.sh "/root/codespace/datasets" "webvision50" 1 0. "uniform" "resnet" 50
```

**Arguments:**
- `<datadir>` - Path to datasets folder, structured as:
    ```
    datasets/
    ├── cifar10/
    │   └── cifar-10-batches-py/
    │       ├── data_batch_1
    │       └── ...
    └── cifar100/
        └── cifar-100-python/
            ├── meta
            └── ...
    ```
- `<dataset>` - Which dataset to use (default: `cifar10`)
- `<seed>` - Random seed (default: `0`)
- `<noise_type>` - Noise type: `uniform` (symmetric) or `flip` (asymmetric) (default: `uniform`)
- `<noise_ratio>` - Fraction of labels to corrupt (default: `0.2`)
- `<net_type>` - Model architecture, see `models/` (default: `resnet`)
- `<depth>` - Model depth, e.g. 32 for ResNet-32 (default: `32`)

>The script `generate_td.sh` calls class `_Dataset` (in `runner.py`) to corrupt the dataset with the given seed, noise_type and noise_ratio, then calls `Runner.train_for_td_computation` to save metadata and acquire training dynamics.

**Output** (saved to `computation4td_seed{seed}/`):

| File | Description |
|------|-------------|
| `model.pth` | Best model (early-stopped) |
| `model.pth.last` | Last epoch model |
| `train_log.csv` | Training log (`epoch`, `train_error`, `train_loss`, `valid_error`, `valid_top5_error`, `valid_loss`) |
| `results_valid.csv` | Per-sample validation results (`index`, `Loss`, `Prediction`, `Confidence`, `Label`) |
| `metadata.pth` | Corruption info: `train_indices`, `valid_indices`, `true_targets`, `assigned_targets`, `label_flipped` |
| `training_dynamics.npz` | Training dynamics: `td` (shape: `[n_samples, n_classes, n_epochs]`) and `labels` (shape: `[n_samples, n_classes]`) |

### STEP 2: Train an LSTM Noise Detector

```sh
# Train a 2-layer LSTM with noisy 0.2 CIFAR-10
CUDA_VISIBLE_DEVICES=0 python train_detector.py --r 0.2 --dataset cifar10 \
    --files_path "./replication/cifar10_resnet32_percmislabeled0.2_uniform/computation4td_seed1"

# Fine-tune a 2-layer LSTM with noisy 0.2 CUB based on an existing detector
CUDA_VISIBLE_DEVICES=0 python train_detector.py --r 0.2 --dataset cub_200_2011 \
    --files_path "./replication/cifar10_resnet34_percmislabeled0.2_uniform/computation4td_seed1" \
    --resume "cifar100_0.3_lstm_detector.pth.tar"
```

Two pre-trained **LSTM** detectors are provided as defaults:
- `cifar10_0.2_lstm_detector.pth.tar` - better for CIFAR-10 tasks
- `cifar100_0.3_lstm_detector.pth.tar` - better for CIFAR-100, Clothing100K, and WebVision50

### STEP 3: Retrain on Cleaned Data

#### 3a. Synthesized datasets (CIFAR-10/100, Tiny ImageNet)

>The script `small_dataset_sym_denoise.sh` calls `Runner.train`, which first invokes `Runner.subset` to detect and remove mislabeled samples. Detection metrics (**AUC** and **mAP**) are reported via `get_order()` from `detector_model/predict.py`. Then the model is retrained on the clean subset.

```sh
small_dataset_sym_denoise.sh <datadir> <dataset> <seed> <noise_ratio> <noise_type> <detector_file> <remove_ratio>

# Denoise symmetric CIFAR-10
detector_files='cifar10_0.2_lstm_detector.pth.tar'
for remove_ratio in 0.15 0.2 0.25; do
    CUDA_VISIBLE_DEVICES=0 ./small_dataset_sym_denoise.sh "/root/codespace/datasets" "cifar10" 1 0.2 "uniform" ${detector_files} ${remove_ratio}
done

# Denoise asymmetric CIFAR-100
for remove_ratio in 0.35 0.4 0.45; do
    CUDA_VISIBLE_DEVICES=0 ./small_dataset_sym_denoise.sh "/root/codespace/datasets" "cifar100" 1 0.4 "asym" ${detector_files} ${remove_ratio}
done
```

#### 3b. Real-world datasets (WebVision50, Clothing100K)

>After ranking all training samples, `Runner.train` selects a cleaner subset to retrain a new model. Requires `training_dynamics.npz` from Step 1 and `<detector_file>` from Step 2.

```sh
large_dataset_denoise.sh <datadir> <dataset> <seed> <detector_file> <remove_ratio>

detector_files='cifar100_0.3_lstm_detector.pth.tar'
remove_ratio=0.2

# Denoise WebVision50
CUDA_VISIBLE_DEVICES=0 ./large_dataset_denoise.sh "/root/codespace/datasets" "webvision50" 1 ${detector_files} ${remove_ratio}
# Denoise Clothing100K
CUDA_VISIBLE_DEVICES=0 ./large_dataset_denoise.sh "/root/codespace/datasets" "clothing100k" 1 ${detector_files} ${remove_ratio}
```

**Arguments:**
- `<detector_file>` - Path to the trained LSTM noise detector from Step 2
- `<remove_ratio>` - Fraction of samples to remove (suspected mislabeled)

**Output** (saved to `prune4retrain_seed{seed}/`):

| File | Description |
|------|-------------|
| `model.pth` | Best model trained on clean subset |
| `model.pth.last` | Last epoch model |
| `train_log.csv` | Training log |
| `results_valid.csv` | Per-sample validation results |

## Paper Replication in Chapter 4.3
---
### Data Debugging to Further Boost SOTA Results

>In Chapter 4.3, we apply a data debugging strategy to further boost SOTA performance. Using a detector trained on noisy CIFAR-100, we first select the most suspicious samples as label noise. We train a new model on the clean part of the dataset. The labels of these samples are then replaced by error-free ones (using ground truth labels for CUB and model predictions for WebVision), namely data debugging.
>
>Implementation details are in `data_debug_dm/cub_200_201/` and `data_debug_dm/mini_webvision/`. Based on the source code of [DivideMix](https://github.com/LiJunnan1992/DivideMix) and [AugDesc](https://github.com/KentoNishi/Augmentation-for-LNL), we mainly modify the datasets' label reading part. We provide the modified dataloaders and trainers for experiments on CUB-200-2011 and mini WebVision.

## Citing
---
If you make use of our work, please cite our paper:

```bibtex
@article{jia2022learning,
  title={Learning from Training Dynamics: Identifying Mislabeled Data Beyond Manually Designed Features},
  author={Jia, Qingrui and Li, Xuhong and Yu, Lei and Bian, Jiang and Zhao, Penghao and Li, Shupeng and Xiong, Haoyi and Dou, Dejing},
  journal={arXiv preprint arXiv:2212.09321},
  year={2022}
}
```

## Credits
---
The implementation is based on [AUM](https://github.com/asappresearch/aum/tree/master/examples/paper_replication) code.
Part of experiments is based on [DivideMix](https://github.com/LiJunnan1992/DivideMix) and [AugDesc](https://github.com/KentoNishi/Augmentation-for-LNL).
Thanks for their brilliant works!
