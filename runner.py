import copy
import datetime
import logging
import os
import random
import shutil
import sys
from collections import OrderedDict

import fire
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import tqdm
from torchvision import datasets, models as tvmodels, transforms

import util
from detector_model import *
from losses import losses
from models import models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed):
    """Set random seed for reproducibility across torch, cuda, and python."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _load_pretrained(model, pretrained_path):
    """Load pretrained weights into model, excluding the final fc layer."""
    try:
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model.state_dict() and 'fc' not in k}
        model.load_state_dict(pretrained_dict, strict=False)
        logging.info("Pretrained model loaded from %s", pretrained_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _Dataset(torch.utils.data.Dataset):
    """
    A wrapper around existing torch datasets to add purposefully mislabeled
    samples and threshold samples.

    :param :obj:`torch.utils.data.Dataset` base_dataset: Dataset to wrap
    :param :obj:`torch.LongTensor` indices: List of indices of base_dataset to include
    :param dict flip_dict: (optional) Mapping of sample indices to their (incorrect) assigned label
    """

    def __init__(self, base_dataset, indices=None, flip_dict=None):
        super().__init__()
        self.dataset = base_dataset
        self.flip_dict = flip_dict or {}
        self.indices = torch.arange(len(self.dataset)) if indices is None else indices
        self.assigned_labels = self.assigned_targets

    @property
    def targets(self):
        """(Hidden) ground-truth labels."""
        if not hasattr(self, "_target_memo"):
            try:
                self.__target_memo = torch.tensor(self.dataset.targets)[self.indices]
            except Exception:
                self.__target_memo = torch.tensor([target for _, target in self.dataset])[self.indices]
        if torch.is_tensor(self.__target_memo):
            return self.__target_memo
        else:
            return torch.tensor(self.__target_memo)

    @property
    def assigned_targets(self):
        """(Potentially incorrect) assigned labels."""
        if not hasattr(self, "_assigned_target_memo"):
            self._assigned_target_memo = self.targets.clone()
            if self.flip_dict is not None:
                for i, idx in enumerate(self.indices.tolist()):
                    if idx in self.flip_dict.keys():
                        self._assigned_target_memo[i] = self.flip_dict[idx]
        return self._assigned_target_memo

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        input, _ = self.dataset[self.indices[index].item()]
        target = self.assigned_labels[index].item()
        return input, target, index


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner(object):
    """
    Main module for running experiments. Can call `load`, `save`, `train`, `test`, etc.

    :param str data: Directory to load data from
    :param str save: Directory to save model/results
    :param str dataset: (cifar10, cifar100, tiny_imagenet, webvision50, clothing100k, cub-200-2011, caltech256)
    :param int num_valid: (default 5000) Size of validation set (from train set, indices determined by seed)
    :param int seed: (default 0) Random seed
    :param int split_seed: (default 0) Seed for train/val split and label flipping
    :param float noise_ratio: (default 0.) Fraction of intentionally mislabeled samples
    :param str noise_type: (uniform, flip) Mislabeling noise model
    :param str loss_type: (default cross-entropy) Loss type
    :param bool oracle_training: (default False) Train only on clean data
    :param str net_type: (resnet, densenet, wide_resnet) Network architecture
    :param **model_args: Additional arguments to pass to the model
    """

    def __init__(self,
                 data,
                 save,
                 dataset="cifar10",
                 num_valid=0,
                 seed=0,
                 split_seed=None,
                 noise_type="uniform",
                 noise_ratio=0.,
                 loss_type="cross-entropy",
                 oracle_training=False,
                 net_type="resnet",
                 depth=18,
                 pretrained=False,
                 **model_args):

        # Save directory
        if not os.path.exists(save):
            os.makedirs(save)
        if not os.path.isdir(save):
            raise Exception('%s is not a dir' % save)
        self.data = data
        self.savedir = save

        # Store configuration
        self.noise_ratio = noise_ratio
        self.noise_type = noise_type
        self.dataset = dataset
        self.net_type = net_type
        self.depth = depth
        self.num_valid = num_valid
        self.split_seed = split_seed if split_seed is not None else seed
        self.seed = seed
        self.loss_func = losses[loss_type]
        self.oracle_training = oracle_training
        self.pretrained = pretrained

        # Seed
        _set_seed(0)

        # Logging
        self.timestring = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logging.basicConfig(
            format='%(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.savedir, 'log-%s.log' % self.timestring)),
            ],
            level=logging.INFO,
        )
        logging.info('Data dir:\t%s' % data)
        logging.info('Save dir:\t%s\n' % save)

        # Dataset info
        self.num_classes = self.test_set.targets.max().item() + 1
        self.num_data = len(self.train_set)
        logging.info(f"\nDataset: {self.dataset}")
        logging.info(f"Num train: {self.num_data}")
        logging.info(f"Num valid: {self.num_valid}")
        logging.info(f"Num classes: {self.num_classes}")
        if self.noise_ratio:
            logging.info(f"Noise type: {self.noise_type}")
            logging.info(f"Flip perc: {self.noise_ratio}\n")
            if self.oracle_training:
                logging.info(f"Training with Oracle Only")

        # Model
        self.model = self._make_model(**model_args)
        logging.info(f"Model type: {self.net_type, self.depth}")
        logging.info(f"Model args:")
        for key, val in model_args.items():
            logging.info(f" - {key}: {val}")
        logging.info(f"Loss type: {loss_type}")
        logging.info("")

    # -------------------------------------------------------------------
    # Model construction
    # -------------------------------------------------------------------

    def _make_model(self, **model_args):
        """Create and return the appropriate model based on dataset and net_type."""
        if self._is_large_dataset():
            return self._make_torchvision_model(pretrained_on_large=True)
        elif self._is_finetune_dataset():
            return self._make_torchvision_model(pretrained_on_large=False)
        else:
            return models[self.net_type](
                num_classes=self.num_classes,
                initial_stride=(2 if "tiny" in self.dataset.lower() else 1),
                **model_args)

    def _is_large_dataset(self):
        """Check if dataset requires ImageNet-scale models."""
        return (self.dataset == "imagenet"
                or "webvision" in self.dataset
                or "clothing" in self.dataset)

    def _is_finetune_dataset(self):
        """Check if dataset requires pretrained + fine-tune models."""
        return "caltech" in self.dataset or "cub" in self.dataset

    def _make_torchvision_model(self, pretrained_on_large=False):
        """Create a torchvision model with optional pretrained weight loading."""
        tv_models = dict((key, val) for key, val in tvmodels.__dict__.items())
        model = tv_models[self.net_type + str(self.depth)](
            pretrained=False, num_classes=self.num_classes)

        # Load pretrained weights
        if pretrained_on_large and self.pretrained:
            _load_pretrained(model, './models/resnet50_pretrained.pth')
        elif not pretrained_on_large:
            pretrained_paths = {
                34: './models/resnet34_pretrained.pth',
                50: './models/resnet50_pretrained.pth',
            }
            if self.depth in pretrained_paths:
                _load_pretrained(model, pretrained_paths[self.depth])
            else:
                logging.info("No pretrained models are locally available")

        # Fix pooling issues for inception
        if pretrained_on_large and "inception" in self.net_type:
            self.avgpool_1a = torch.nn.AdaptiveAvgPool2d((1, 1))

        return model

    # -------------------------------------------------------------------
    # Dataset construction
    # -------------------------------------------------------------------

    def _get_transforms(self):
        """Return (train_transforms, test_transforms) for the current dataset."""
        if self.big_model:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            crop_size = 227 if "inception" in self.net_type else 224
            test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif self.dataset == "tiny_imagenet":
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                test_transforms,
            ])
        elif self.dataset in ("cifar10", "cifar100"):
            normalize = transforms.Normalize(
                mean=[0.4914, 0.4824, 0.4467], std=[0.2471, 0.2435, 0.2616])
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                test_transforms,
            ])
        elif self.dataset == "mnist":
            normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            test_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            train_transforms = test_transforms
        else:
            raise ValueError(f"Unknown dataset {self.dataset}")

        return train_transforms, test_transforms

    def _make_datasets(self):
        """Build train/valid/test dataset splits with optional noise injection."""
        # Determine dataset class
        try:
            dataset_cls = getattr(datasets, self.dataset.upper())
            self.big_model = False
        except Exception:
            dataset_cls = datasets.ImageFolder
            self.big_model = "tiny" not in self.dataset.lower()

        # Get dataset constants
        if dataset_cls == datasets.ImageFolder:
            tmp_set = dataset_cls(root=os.path.join(self.data, "train"))
        else:
            tmp_set = dataset_cls(root=self.data, train=True, download=False)
            if self.dataset.upper() == 'CIFAR10':
                tmp_set.target = tmp_set.targets
        num_train = len(tmp_set) - self.num_valid
        num_valid = self.num_valid
        num_classes = int(max(tmp_set.targets)) + 1
        self.num_classes = num_classes

        # Create train/valid split
        _set_seed(self.split_seed)
        train_indices, valid_indices = torch.randperm(num_train + num_valid).split(
            [num_train, num_valid])

        # Generate label noise
        flip_dict = self._generate_noise(tmp_set, train_indices, num_classes)

        # Reset the seed for dataset/initializations
        _set_seed(self.split_seed)

        # Build transforms and datasets
        train_transforms, test_transforms = self._get_transforms()
        self._build_dataset_splits(dataset_cls, train_transforms, test_transforms,
                                   train_indices, valid_indices, flip_dict)

    def _generate_noise(self, tmp_set, train_indices, num_classes):
        """Generate label flip dictionary based on noise_type and noise_ratio."""
        flip_dict = {}
        if not self.noise_ratio:
            return flip_dict

        # Generate noisy labels from random transitions
        transition_matrix = torch.eye(num_classes)
        if self.noise_type == "uniform":
            transition_matrix.mul_(1 - self.noise_ratio * (num_classes / (num_classes - 1)))
            transition_matrix.add_(self.noise_ratio / (num_classes - 1))
        elif self.noise_type == "flip":
            source_classes = torch.arange(num_classes)
            target_classes = (source_classes + 1).fmod(num_classes)
            transition_matrix.mul_(1 - self.noise_ratio)
            transition_matrix[source_classes, target_classes] = self.noise_ratio
        else:
            raise ValueError(f"Unknown noise type {self.noise_type}")

        true_targets = (torch.tensor(tmp_set.targets) if hasattr(tmp_set, "targets") else
                        torch.tensor([target for _, target in self]))
        transition_targets = torch.distributions.Categorical(
            probs=transition_matrix[true_targets, :]).sample()

        if not self.oracle_training:
            flip_indices = torch.nonzero(transition_targets != true_targets).squeeze(-1)
            flip_targets = transition_targets[flip_indices]
            for index, target in zip(flip_indices, flip_targets):
                flip_dict[index.item()] = target.item()
        else:
            # In the oracle setting, don't add transitions
            oracle_indices = torch.nonzero(transition_targets == true_targets).squeeze(-1)
            train_indices[:] = torch.from_numpy(
                np.intersect1d(oracle_indices.numpy(), train_indices.numpy())).long()

        return flip_dict

    def _build_dataset_splits(self, dataset_cls, train_transforms, test_transforms,
                              train_indices, valid_indices, flip_dict):
        """Create _train_set_memo, _valid_set_memo, _test_set_memo."""
        if dataset_cls == datasets.ImageFolder:
            self._train_set_memo = _Dataset(
                dataset_cls(root=os.path.join(self.data, "train"), transform=train_transforms),
                flip_dict=flip_dict, indices=train_indices)

            if os.path.exists(os.path.join(self.data, "test")):
                self._valid_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "val"), transform=test_transforms))
                self._test_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "test"), transform=test_transforms))
            else:
                self._valid_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "train"), transform=test_transforms),
                    indices=valid_indices) if len(valid_indices) else None
                self._test_set_memo = _Dataset(
                    dataset_cls(root=os.path.join(self.data, "val"), transform=test_transforms))
        else:
            self._train_set_memo = _Dataset(
                dataset_cls(root=self.data, train=True, transform=train_transforms),
                flip_dict=flip_dict, indices=train_indices)
            self._valid_set_memo = _Dataset(
                dataset_cls(root=self.data, train=True, transform=test_transforms),
                indices=valid_indices) if len(valid_indices) else None
            self._test_set_memo = _Dataset(
                dataset_cls(root=self.data, train=False, transform=test_transforms))

    # -------------------------------------------------------------------
    # Dataset properties (lazy loading)
    # -------------------------------------------------------------------

    @property
    def test_set(self):
        if not hasattr(self, "_test_set_memo"):
            self._make_datasets()
        return self._test_set_memo

    @property
    def train_set(self):
        if not hasattr(self, "_train_set_memo"):
            self._make_datasets()
        return self._train_set_memo

    @property
    def valid_set(self):
        if not hasattr(self, "_valid_set_memo"):
            self._make_datasets()
        return self._valid_set_memo

    # -------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------

    def done(self):
        "Break out of the runner"
        return None

    def load(self, target_model=None, save=None, suffix=""):
        """
        Load a previously saved model state dict.

        :param str save: (optional) Folder to load from. Defaults to current runner's save dir.
        :param str suffix: (optional) Model file suffix (e.g. ".last").
        """
        if target_model is None:
            target_model = self.model
        save = save or self.savedir
        state_dict = torch.load(os.path.join(save, f"model.pth{suffix}"),
                                map_location=torch.device('cpu'))
        target_model.load_state_dict(state_dict, strict=False)
        return self

    def save(self, target_model=None, save=None, suffix=""):
        """
        Save the current state dict.

        :param str save: (optional) Folder to save to. Defaults to current runner's save dir.
        :param str suffix: (optional) A suffix to append to the save name.
        """
        if target_model is None:
            target_model = self.model
        save = save or self.savedir
        torch.save(target_model.state_dict(), os.path.join(save, f"model.pth{suffix}"))
        return self

    # -------------------------------------------------------------------
    # Subset / Denoising
    # -------------------------------------------------------------------

    def subset(self, perc, td_files=None, detector_files=None):
        """
        Use only a subset of the training set.
        If training dynamics file is supplied, drop samples with the lowest
        probability of being predicted as noise. Otherwise, drop samples at random.

        :param float perc: What percentage of the set to remove
        :param str td_files: Training dynamics file's path
        :param str detector_files: Trained detector file's path
        """
        if td_files is None:
            _set_seed(self.seed)
            order = torch.randperm(len(self.train_set))
        else:
            training_dynamics = np.load(os.path.join(td_files, "training_dynamics.npz"))['td']
            td = training_dynamics[:, :, 0]  # extract only ground truth as input for transferability
            td = np.expand_dims(td, axis=2)
            td = torch.tensor(td)
            print('Using input type with shape of', td.shape)
            try:
                label_flip = torch.load(os.path.join(td_files, "metadata.pth"))['label_flipped']
            except Exception:
                label_flip = torch.ones((len(td)))

            noise_detector = LSTM(in_dim=td.shape[-1], hidden_dim=64, n_layer=2)
            noise_detector.load_state_dict(torch.load(detector_files)['state_dict'])
            if torch.cuda.is_available():
                noise_detector.cuda()

            dataset = torch.utils.data.TensorDataset(td, label_flip)
            loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
            order, auc, map = get_order(loader, noise_detector)

        num_samples = int(len(order) * (1 - perc))
        self.train_set.indices = self.train_set.indices[order[:num_samples]]
        self.train_set.assigned_labels = self.train_set.assigned_labels[order[:num_samples]]

        logging.info(f"Reducing training set from {len(order)} to {len(self.train_set)}")
        return self

    # -------------------------------------------------------------------
    # Test / Evaluate
    # -------------------------------------------------------------------

    def test(self, model=None, split="test", batch_size=512, td=None,
             dataset=None, epoch=None, num_workers=4):
        """Testing script."""
        stats = ['error', 'top5_error', 'loss']
        meters = [util.AverageMeter() for _ in stats]
        result_class = util.result_class(stats)

        # Get model
        if model is None:
            model = self.model
            if torch.cuda.is_available():
                model = model.cuda()
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model).cuda()

        # Get dataset/loader
        if dataset is None:
            try:
                dataset = getattr(self, f"{split}_set")
            except Exception:
                raise ValueError(f"Invalid split '{split}'")

        loader = tqdm.tqdm(
            torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=num_workers),
            desc=split.title())

        # Storage for results
        all_losses = []
        all_confs = []
        all_preds = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for inputs, targets, indices in loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = model(inputs)
                batch_losses = self.loss_func(outputs, targets, reduction="none")
                topk = 5 if self.num_classes > 5 else 1
                confs, preds = outputs.topk(topk, dim=-1, largest=True, sorted=True)
                is_correct = preds.eq(targets.unsqueeze(-1)).float()
                loss = batch_losses.mean()
                error = 1 - is_correct[:, 0].mean()
                top5_error = 1 - is_correct.sum(dim=-1).mean()

                # Measure and record stats
                cur_batch_size = inputs.size(0)
                stat_vals = [error.item(), top5_error.item(), loss.item()]
                for stat_val, meter in zip(stat_vals, meters):
                    meter.update(stat_val, cur_batch_size)

                all_losses.append(batch_losses.cpu())
                all_confs.append(confs[:, 0].cpu())
                all_preds.append(preds[:, 0].cpu())
                all_targets.append(targets.cpu())

                # Log stats
                res = dict((name, f"{meter.val:.3f} ({meter.avg:.3f})")
                           for name, meter in zip(stats, meters))
                loader.set_postfix(**res)

        # Save the outputs
        pd.DataFrame({
            "Loss": torch.cat(all_losses).numpy(),
            "Prediction": torch.cat(all_preds).numpy(),
            "Confidence": torch.cat(all_confs).numpy(),
            "Label": torch.cat(all_targets).numpy(),
        }).to_csv(os.path.join(self.savedir, f"results_{split}.csv"), index_label="index")

        return result_class(*[meter.avg for meter in meters])

    # -------------------------------------------------------------------
    # Training dynamics generation
    # -------------------------------------------------------------------

    def generate_training_dynamics(self, model=None, batch_size=512, td=None, num_workers=4):
        """Record model predictions on full training set for training dynamics."""
        if model is None:
            model = self.model
            if torch.cuda.is_available():
                model = model.cuda()
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model).cuda()

        full_train_set = self.full_train_set
        loader = tqdm.tqdm(
            torch.utils.data.DataLoader(full_train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers),
            desc="Generating training dynamics")

        model.eval()
        with torch.no_grad():
            for inputs, targets, indices in loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = model(inputs)
                softmax = torch.nn.Softmax(dim=1)
                outputs = softmax(outputs).detach().cpu().numpy()

                for j in range(len(indices)):
                    idx = indices[j].item()
                    idx_probs = td.get(idx, [])
                    idx_probs.append(outputs[j])
                    td[idx] = idx_probs

        return td

    # -------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------

    def train_for_td_computation(self, num_epochs=200, batch_size=128,
                                 lr=0.1, wd=5e-4, momentum=0.9, **kwargs):
        """
        Training script specifically for training dynamics computation.
        Uses smaller batch size to reduce memorization.
        """
        return self.train(num_epochs=num_epochs, batch_size=batch_size,
                          test_at_end=False, lr=lr, wd=wd, momentum=momentum,
                          lr_drops=[0.5, 0.75], **kwargs)

    def train(self, num_epochs=200, batch_size=256, test_at_end=True,
              lr=0.1, wd=1e-4, momentum=0.9, lr_drops=[0.5, 0.75],
              td_files=False, remove_ratio=0, detector_files=None,
              rand_weight=False, **kwargs):
        """
        Training script.

        :param int num_epochs: (default 200)
        :param int batch_size: (default 256)
        :param float lr: Learning rate
        :param float wd: Weight decay
        :param float momentum: Momentum
        :param list lr_drops: When to drop lr (as percentage of total epochs)
        :param str td_files: (optional) Path to load training dynamics from
        :param bool rand_weight: (optional) Use rectified normal random weighting
        """
        # Model setup
        model = self.model
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model).cuda()

        # Optimizer and scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd,
                                    momentum=momentum, nesterov=True)
        milestones = [int(lr_drop * num_epochs) for lr_drop in (lr_drops or [])]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        cudnn.benchmark = True
        logging.info(f"\nOPTIMIZER:\n{optimizer}")
        logging.info(f"SCHEDULER:\n{milestones}")
        self.good_probs = None

        # Save metadata about train set (which labels were flipped, etc.)
        metadata = OrderedDict()
        metadata["train_indices"] = self.train_set.indices
        metadata["valid_indices"] = (self.valid_set.indices if self.valid_set is not None
                                     else torch.tensor([], dtype=torch.long))
        metadata["true_targets"] = self.train_set.targets
        metadata["assigned_targets"] = self.train_set.assigned_targets
        metadata["label_flipped"] = torch.ne(self.train_set.targets, self.train_set.assigned_targets)
        torch.save(metadata, os.path.join(self.savedir, "metadata.pth"))

        self.full_train_set = copy.deepcopy(self.train_set)

        # Optional: subset training data using detector
        if td_files:
            self.subset(perc=remove_ratio, td_files=td_files, detector_files=detector_files)
            logging.info(f"(Num samples remained to training phase: {len(self.train_set)})")
        elif rand_weight:
            logging.info("Rectified Normal Random Weighting")
        else:
            logging.info("Standard weighting")

        # Balance batch_size when removing samples
        batch_size = int(batch_size * (1 - remove_ratio))

        # Training loop
        results = []
        td = {}
        best_error = 1

        for epoch in range(num_epochs):
            train_results = self.train_epoch(
                model=model, optimizer=optimizer, epoch=epoch,
                num_epochs=num_epochs, batch_size=batch_size,
                good_probs=self.good_probs, rand_weight=rand_weight, **kwargs)

            if self.valid_set is not None:
                valid_results = self.test(model=model, split="valid",
                                         batch_size=batch_size, epoch=epoch, **kwargs)
            else:
                valid_results = self.test(model, split="test",
                                         batch_size=batch_size, epoch=epoch, **kwargs)

            # Generate training dynamics for full train set
            td = self.generate_training_dynamics(model, batch_size=batch_size, td=td, **kwargs)
            scheduler.step()

            # Save best model
            if self.valid_set is not None:
                self.save()
            elif best_error > valid_results.error:
                best_error = valid_results.error
                logging.info('New best error: %.4f' % valid_results.error)
                self.save()

            # Log results
            logging.info(f"\nTraining {repr(train_results)}")
            logging.info(f"\nValidation {repr(valid_results)}")
            logging.info('')
            results.append(
                OrderedDict([("epoch", f"{epoch + 1:03d}"),
                             *[(f"train_{field}", val) for field, val in train_results.items()],
                             *[(f"valid_{field}", val) for field, val in valid_results.items()]]))
            pd.DataFrame(results).set_index("epoch").to_csv(
                os.path.join(self.savedir, "train_log.csv"))

        # Test at end (last epoch)
        if test_at_end and self.valid_set is not None:
            test_results = self.test(model=model, **kwargs)
            logging.info(f"\nTest (no early stopping) {repr(test_results)}")
            shutil.copyfile(os.path.join(self.savedir, "results_test.csv"),
                            os.path.join(self.savedir, "results_test_noearlystop.csv"))
            results.append(
                OrderedDict([(f"test_{field}", val) for field, val in test_results.items()]))
            pd.DataFrame(results).set_index("epoch").to_csv(
                os.path.join(self.savedir, "train_log.csv"))

        # Load best model
        self.save(suffix=".last")
        self.load()

        # Test at end (best epoch)
        if test_at_end and self.valid_set is not None:
            test_results = self.test(model=model, **kwargs)
            logging.info(f"\nEarly Stopped Model Test {repr(test_results)}")
            results.append(
                OrderedDict([(f"test_best_{field}", val) for field, val in test_results.items()]))
        pd.DataFrame(results).set_index("epoch").to_csv(
            os.path.join(self.savedir, "train_log.csv"))

        # Save training dynamics
        util.transfor_and_save(self.savedir, td, self.train_set.assigned_labels)
        logging.info("Saved Training Dynamics to %s" % self.savedir)

        return self

    def train_epoch(self, model, optimizer, epoch, num_epochs, batch_size=256,
                    num_workers=4, good_probs=None, rand_weight=False):
        """Run a single training epoch."""
        stats = ["error", "loss"]
        meters = [util.AverageMeter() for _ in stats]
        result_class = util.result_class(stats)

        train_set = self.train_set
        loader = tqdm.tqdm(
            torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=num_workers),
            desc=f"Train (Epoch {epoch + 1}/{num_epochs})")

        model.train()
        for inputs, targets, indices in loader:
            optimizer.zero_grad()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            # Forward pass
            outputs = model(inputs)
            batch_losses = self.loss_func(outputs, targets, reduction="none")
            preds = outputs.argmax(dim=-1)

            # Compute loss weights
            if good_probs is not None:
                if torch.cuda.is_available():
                    good_probs = self.good_probs.cuda()
                weights = good_probs[indices.to(good_probs.device)]
                weights = weights.div(weights.sum())
            elif rand_weight:
                weights = torch.randn(targets.size(), dtype=outputs.dtype,
                                      device=outputs.device).clamp_min_(0)
                weights = weights.div(weights.sum().clamp_min_(1e-10))
            else:
                weights = torch.ones(targets.size(), dtype=outputs.dtype,
                                     device=outputs.device).div_(targets.numel())

            # Backward pass and update
            loss = torch.dot(weights, batch_losses)
            error = torch.ne(targets, preds).float().mean()
            loss.backward()
            optimizer.step()

            # Measure and record stats
            cur_batch_size = outputs.size(0)
            stat_vals = [error.item(), loss.item()]
            for stat_val, meter in zip(stat_vals, meters):
                meter.update(stat_val, cur_batch_size)

            res = dict(
                (name, f"{meter.val:.3f} ({meter.avg:.3f})")
                for name, meter in zip(stats, meters))
            loader.set_postfix(**res)

        return result_class(*[meter.avg for meter in meters])


if __name__ == "__main__":
    fire.Fire(Runner)
