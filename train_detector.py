import shutil
import os
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import argparse

parser = argparse.ArgumentParser(description='Noise Detector Training')
# parser.add_argument('--r', default=0.2, type=float, description='noise ratio')
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--resume', default=None, type=str, help='checkpoint for fine-tuning')
parser.add_argument('--files_path', type=str, help='metadata and training dynamics path')
args = parser.parse_args()

def save_checkpoint(savedir, state, is_best):
    if is_best:
        filepath = savedir+'.pth.tar'
        torch.save(state, filepath)

class LSTM(nn.Module):
    def __init__(self,hidden_dim=64,n_layer=2,in_dim=1,n_class=2,dropout=0):
        super(LSTM,self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.in_dim = in_dim
        self.classifier = nn.Linear(hidden_dim,n_class) 
        self.lstm = nn.LSTM(self.in_dim,hidden_dim,n_layer,batch_first=True,dropout=self.dropout, bidirectional=False)

    def forward(self,x):        
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.classifier(out)
        return out

def main():
    # load dataset for training
    td = np.load(os.path.join(args.files_path,"training_dynamics.npz"))['td'][:,:,0]  #extract only ground turth
    td = np.expand_dims(td, axis=2)

    is_noisy = torch.load(os.path.join(args.files_path,"metadata.pth"))['label_flipped'].to(dtype=torch.int64)
    td = torch.tensor(td,dtype=torch.float)
    print('Using input type with shape of', td.shape)
    
    # define model
    net = LSTM().cuda()
    print('Training detector instanced by',net.__class__.__name__)
        
    # load checkpoint or train from scratch
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
            
    train_x, test_x, train_y, test_y = train_test_split(td, is_noisy, test_size=0.1, random_state=42)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.1)
    max_epoch = 10
    best_prec = 0
        
    for epoch in range(max_epoch):
        net.train()
        loss_sigma = 0.0  #
        correct = 0.0
        total = 0.0
        for i,(train_data,train_label) in enumerate(train_dataloader):
            train_data,train_label = Variable(train_data).cuda(),Variable(train_label).cuda()
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

        # evaluation on test set
        net.eval()
        conf_matrix = np.zeros((2,2))
        with torch.no_grad():
            for it,(test_data,test_label) in enumerate(test_dataloader):
                test_data,test_label = Variable(test_data).cuda(),Variable(test_label).cuda()
                test_out = net(test_data)

                _, predicted = torch.max(test_out.data, 1)
                for i in range(predicted.shape[0]):
                    conf_matrix[test_label[i],predicted[i]]+=1

        test_acc = np.diag(conf_matrix).sum()/np.sum(conf_matrix)
        is_best = test_acc > best_prec
        best_prec = max(test_acc, best_prec)
        
        # save checkpoint
        save_checkpoint(
            savedir='%s_%.1f_lstm_detector'%(args.dataset,args.r),
            state={
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            }, 
            is_best=is_best)

if __name__ == "__main__":
    main()
