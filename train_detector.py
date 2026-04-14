import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import argparse

from detector_model.lstm import LSTM

parser = argparse.ArgumentParser(description='Noise Detector Training')
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--resume', default=None, type=str, help='checkpoint for fine-tuning')
parser.add_argument('--files_path', type=str, help='metadata and training dynamics path')
args = parser.parse_args()


def save_checkpoint(savedir, state, is_best):
    if is_best:
        filepath = savedir + '.pth.tar'
        torch.save(state, filepath)


def main():
    # Load dataset for training
    td = np.load(os.path.join(args.files_path, "training_dynamics.npz"))['td'][:, :, 0]  # extract only ground truth
    td = np.expand_dims(td, axis=2)

    is_noisy = torch.load(os.path.join(args.files_path, "metadata.pth"))['label_flipped'].to(dtype=torch.int64)
    td = torch.tensor(td, dtype=torch.float)
    print('Using input type with shape of', td.shape)

    # Define model
    net = LSTM(in_dim=td.shape[-1]).cuda()
    print('Training detector instanced by', net.__class__.__name__)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.1)

    # Load checkpoint or train from scratch
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Train/test split
    train_x, test_x, train_y, test_y = train_test_split(td, is_noisy, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    max_epoch = 10
    best_prec = 0

    for epoch in range(max_epoch):
        # Training
        net.train()
        loss_sigma = 0.0
        correct = 0.0
        total = 0.0
        for i, (train_data, train_label) in enumerate(train_dataloader):
            train_data, train_label = train_data.cuda(), train_label.cuda()
            out = net(train_data)

            loss = criterion(out, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(out.data, 1)
            total += train_label.size(0)
            correct += (predicted == train_label).squeeze().sum().cpu().numpy()
            loss_sigma += loss.item()

        print("Training: Epoch[{:0>3}/{:0>3}]  Loss: {:.4f} Acc:{:.2%}".format(
            epoch + 1, max_epoch, loss_sigma, correct / total))

        # Evaluation on test set
        net.eval()
        conf_matrix = np.zeros((2, 2))
        with torch.no_grad():
            for it, (test_data, test_label) in enumerate(test_dataloader):
                test_data, test_label = test_data.cuda(), test_label.cuda()
                test_out = net(test_data)

                _, predicted = torch.max(test_out.data, 1)
                for i in range(predicted.shape[0]):
                    conf_matrix[test_label[i], predicted[i]] += 1

        test_acc = np.diag(conf_matrix).sum() / np.sum(conf_matrix)
        is_best = test_acc > best_prec
        best_prec = max(test_acc, best_prec)

        # Save checkpoint
        save_checkpoint(
            savedir='%s_%.1f_lstm_detector' % (args.dataset, args.r),
            state={
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec': best_prec,
                'optimizer': optimizer.state_dict(),
            },
            is_best=is_best)


if __name__ == "__main__":
    main()
