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
- pytorch >=1.7.1
- torchvision >= 0.4
- scikit-learn
- numpy
- pandas

## Paper Replication in Chapters 4.1 and 4.2
---
### Datasets
We run experiments on 5 **small** datasets...
- cifar10
- cifar100
- tiny imagenet
- cub 200 2011
- caltech256

... and 2 **large** datasets
- webvision50(subset of webvision50)
- clothing100k(subset of clothing 1M)

We use the same subset as [AUM](https://github.com/asappresearch/aum/tree/master/examples/paper_replication) for these two subsets. Click [Here](https://drive.google.com/file/d/1rr2nvnnBMsbo1qcU3i3urJsDw86PJ9tR/view?usp=sharing) to download and untar the file to access Cub-200-2011 and Caltech256. 

### Replication Steps: 
1. Acquisition of metadata and training dynamics (short for td) for manually corrupted or real-world datasets.
2. Training an LSTM model as a detector
3. Retraining new model on clean data, including two parts as follow:
- Metrics of label noise detection on synthesized datasets (CIFAR-10/100, Tiny ImageNet) and Retraining new model on clean data
- Less overfitting towards noisy labels on real-world datasets (WebVision50 and Clothing100K)

### STEP1: Acquisition of metadata and training dynamics (short for td) for manually corrupted or real-world datasets.

```sh
generate_td.sh <datadir> <dataset> <seed> <noise_ratio> <noise_type> <net_type> <depth> <result_save_path>

# run to get td for small datasets [no manual corruption]
CUDA_VISIBLE_DEVICES=0 generate_td.sh "/root/codespace/datasets" "cifar10" 1 0. "uniform" "resnet" 32 "./replication"
# run to get td for small datasets [uniform 0.2 noisy]
CUDA_VISIBLE_DEVICES=0 generate_td.sh "/root/codespace/datasets" "tiny_imagenet" 1 0.2 "uniform" "resnet" 32 "./replication"
# run to get td for large datasets [noise_ratio and noise_type are mute]
CUDA_VISIBLE_DEVICES=0 generate_td.sh "/root/codespace/datasets" "webvision50" 1 0. "uniform" "resnet" 50 "./replication"
```

The arguments:
- `<datadir>` - a path of datasets folder
    be like:
    ```tree
    |-- datasets
        |-- cifar10
        |   |-- cifar-10-batches-py
        |   |   |-- data_batch_1
        |   |   |-- ...
        |-- cifar100
        |   |-- cifar-100-python
        |   |   |-- meta
        |   |   |-- ...
    ```
- `<dataset>` - default = `cifar10` indicates which dataset to use
- `<seed>` - default = `0` indicates the random seed
- `<noise_type>` - default = `uniform` indicates which type of noise, `uniform` means `symmetric` and `flip` means `asymmetric`
- `<noise_ratio>` - default = `0.2` indicates how many labels are corrupted
- `<net_type>` - default = `resnet` indicates which model to apply, can be modified in /models
- `<depth>` - default = `32` indicates depth of model. For example, the depth of resnet32 is 32.
- `<result_save_path>` - default = 'replication' indicates where to save experiments

>The script `generate_td.sh` calls class `_Dataset` to corrupt dataset chosen with certain seed,noise_type and noise_ratio and calls function `Runner.train_for_td_computation` which saves the metadata(corruption information) and train a classification model to acquire training dynamics. 
>Both of them can be found in `runner.py`

After running this, the code will save all the following in one folder named `computation4td_seed{seed}`.

- model.pth --> best model
- model.pth.last --> last model
- train_log.csv --> record of the training process
```
| epoch | train_error | train_loss | valid_error | valid_top5_error | valid_loss |
```
- results_valid.csv --> sample-wised validation results
```
| index | Loss | Prediction | Confidence | Label |
```
- metadata.pth --> corruption information
```
OrderedDict([
    ('train_indices',tensor([45845,  ..., 8475])),
    ('valid_indices',tensor([], dtype=torch.int64)),
    ('true_targets', tensor([56,  ..., 67])),
    ('label_flipped',tensor([False,  ..., True]))])
```
- training_dynamics.npz --> saved td file
```dict
{
    td:{}, - type: array, shape: 
                            [number of samples in training,
                            N(GT + top-(N-1) average probabilities among all classes of all epochs),
                            training length]
    labels:{}, - type: array, shape:
                            [number of samples in training,
                            N(labels of GT + top-(N-1) probabilities)]
}
```

### STEP2: Training an LSTM model as a detector
```sh
# train a 2-layer lstm with noisy 0.2 cifar10 
CUDA_VISIBLE_DEVICES=0 python train_detector.py --r 0.2 --dataset cifar10 --files_path "./replication/cifar10_resnet32_percmislabeled0.2_uniform/computation4td_seed1"
# fine tuning a 2-layer lstm with noisy 0.2 cub based on cifar10_0.2_lstm_detector.pth.tar
CUDA_VISIBLE_DEVICES=0 python train_detector.py --r 0.2 --dataset cub_200_2011 --files_path "./replication/cifar10_resnet34_percmislabeled0.2_uniform/computation4td_seed1" --resume "cifar100_0.3_lstm_detector.pth.tar"
```

2 common **LSTM** models as default, each one is OK for both CIFAR-10 or CIFAR-100 task, but is better when corresponds to task:
- `cifar10_0.2_lstm_detector.pth.tar` better for cifar10
- `cifar100_0.3_lstm_detector.pth.tar` better for cifar100 and used in noide detection of Clothing100K and Webvision50.


### STEP3: Retraining new model on clean data

#### Metrics of label noise detection on synthesized datasets (CIFAR-10/100, Tiny ImageNet) and Retraining new model on clean data

>The script `small_dataset_sym_denoise.sh` calls the function `runner.Runner.train`. This function begins with `runner.Runner.subset` that detects the mislabeled samples and divides the original training set into clean and noisy sets. Meanwhile, the metrics of **ROC**, and**mAP** of identifying mislabeled samples are also reported by calling `get_order` from `from detector_models.predict`.

> After detection, the function `runner.Runner.train` uses the clean set to train a new model. (The amount of this part depends on `remove_ratio`.) We note that the function `runner.Runner.train` requires `metadata.pth`, `training_dynamics.npz` and `<detector_files>`, where the first two come from Step 1 and the third comes from Step 2. 

```sh
small_dataset_sym_denoise.sh <datadir> <dataset> <seed> <noise_ratio> <noise_type> <result_save_path> <detector_file> <remove_ratio>
# run to detect-divide target dataset and retrain the model
detector_files='lstm_cifar10_20.pth.tar'
# run to denoise sym cifar10
for remove_ratio in 0.15 0.2 0.25
do
CUDA_VISIBLE_DEVICES=0 small_dataset_sym_denoise.sh "/root/codespace/datasets" "cifar10" 1 0.2 "uniform" "./replication" ${detector_files} ${remove_ratio}
done
# run to denoise asym cifar100
for remove_ratio in 0.35 0.4 0.45
do
CUDA_VISIBLE_DEVICES=0 small_dataset_sym_denoise.sh "/root/codespace/datasets" "cifar100" 1 0.4 "asym" "./replication" ${detector_files} ${remove_ratio}
done
```

#### Less overfitting towards noisy labels on real-world datasets (WebVision50 and Clothing100K)

> After ranking all training samples, the function `runner.Runner.train` selects a more clean set to train a new model. (The amount of this part depends on `remove_ratio`.) We note that the function `runner.Runner.train` requires `training_dynamics.npz` and `<detector_files>`, where the first two come from Step 1 and the third comes from Step 2. 
After running this, the code will save all the followings in another folder named `{net_name}_prune4retrain_seed{seed}`.

```sh
large_dataset_denoise.sh <datadir> <dataset> <seed> <result_save_path> <detector_file> <remove_ratio>
# run to detect-divide target dataset and retrain the model
detector_files='mono_lstm_cifar100_30.pth.tar'
# run to denoise WebVision50
CUDA_VISIBLE_DEVICES=0 large_dataset_denoise.sh "/root/codespace/datasets" "webvision50" 1 "./replication" ${detector_files} ${remove_ratio}
# run to denoise Clothing100K
CUDA_VISIBLE_DEVICES=0 large_dataset_denoise.sh "/root/codespace/datasets" "clothing100k" 1 "./replication" ${detector_files} ${remove_ratio}
```

Arguments:
- `<detector_files>` - noise detector instanced by 2-layers LSTM trained in Step2.
- `<remove_ratio>` - the ratio of samples we removed, which are believed to be noisy/mislabeled samples.

Output: 
- model.pth --> best model trained by clean part
- model.pth.last --> last model trained by clean part
- train_log.csv --> record of the training process
```
| epoch | train_error | train_loss | valid_error | valid_top5_error | valid_loss |
```
- results_valid.csv --> sample-wised validation results
```
| index | Loss | Prediction | Confidence | Label |
```

## Paper Replication in Chapter 4.3
---
### Perform data degging to further boost SOTA results
>In Chapter 4.3, we apply a data degging strategy to further boost SOTA performance. Using a detector trained by noisy CIFAR-100, we first select
several samples with the most suspicion as label noise. We train a new **MODEL** on the clean part of the dataset. The labels of these samples are then replaced by error-free ones (using ground truth labels for CUB and **MODEL** prediction for Webvision), namely data debugging, recorded in cub_200_2011/ and mini_webvision/.
For replication, the only difference we made is the label of datasets. Based on the source code and instructions of [DivideMix](https://github.com/LiJunnan1992/DivideMix) and [AugDesc](https://github.com/KentoNishi/Augmentation-for-LNL), we mainly modify the datasets' labels reading part. We provide the modified dataloader.py and trainer for experiments on sym CUB-200-2011 and mini Webvision to boost DivideMix.

## Citing
---
>If you make use of our work, please cite our paper:

```
@article{jia2022learning,
  title={Learning from Training Dynamics: Identifying Mislabeled Data Beyond Manually Designed Features},
  author={Jia, Qingrui and Li, Xuhong and Yu, Lei and Bian, Jiang and Zhao, Penghao and Li, Shupeng and Xiong, Haoyi and Dou, Dejing},
  journal={arXiv preprint arXiv:2212.09321},
  year={2022}
}
```

## Credits
The implementation is based on [AUM](https://github.com/asappresearch/aum/tree/master/examples/paper_replication) code.
Part of experiments is based on [DivideMix](https://github.com/LiJunnan1992/DivideMix) and [AugDesc](https://github.com/KentoNishi/Augmentation-for-LNL)
Thanks for their brilliant work!