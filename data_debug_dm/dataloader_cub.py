from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

import torchvision
            
# def unpickle(file):
#     import _pickle as cPickle
#     with open(file, 'rb') as fo:
#         dict = cPickle.load(fo, encoding='latin1')
#     return dict

class cub_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log=''): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        # self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            # if dataset=='cifar10':                
            #     test_dic = unpickle('%s/test_batch'%root_dir)
            #     self.test_data = test_dic['data']
            #     self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            #     self.test_data = self.test_data.transpose((0, 2, 3, 1))  
            #     self.test_label = test_dic['labels']
            # elif dataset=='cifar100':
            #     test_dic = unpickle('%s/test'%root_dir)
            #     self.test_data = test_dic['data']
            #     self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            #     self.test_data = self.test_data.transpose((0, 2, 3, 1))  
            #     self.test_label = test_dic['fine_labels']      
            assert dataset == "cub_200_2011"
            self.test_dataset = torchvision.datasets.ImageFolder("%s/val"%root_dir,transform = self.transform)

                
        else:    
            # train_data=[]
            # train_label=[]
            # if dataset=='cifar10': 
            #     for n in range(1,6):
            #         dpath = '%s/data_batch_%d'%(root_dir,n)
            #         data_dic = unpickle(dpath)
            #         train_data.append(data_dic['data'])
            #         train_label = train_label+data_dic['labels']
            #     train_data = np.concatenate(train_data)
            # elif dataset=='cifar100':    
            #     train_dic = unpickle('%s/train'%root_dir)
            #     train_data = train_dic['data']
            #     train_label = train_dic['fine_labels']
            # train_data = train_data.reshape((50000, 3, 32, 32))
            # train_data = train_data.transpose((0, 2, 3, 1))
                       
            assert dataset == "cub_200_2011"
            self.train_dataset = torchvision.datasets.ImageFolder("%s/train"%root_dir,transform = self.transform)       
            train_label = self.train_dataset.targets
    
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
                print('Noisy samples: 'sum(torch.ne(torch.tensor(noise_label),torch.tensor(train_label))))
                print("Load noisy labels from %s ..."%noise_file)        
            else:    #inject noise   
                noise_label = []
                idx = list(range(len(train_label)))
                random.shuffle(idx)
                num_noise = int(self.r*len(train_label))            
                noise_idx = idx[:num_noise]
                for i in range(len(train_label)):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            noiselabel = random.randint(0,199)
                            noise_label.append(noiselabel)
                        # elif noise_mode=='asym':   
                        #     noiselabel = self.transition[train_label[i]]
                        #     noise_label.append(noiselabel)                    
                    else:    
                        noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w")) 
                # raise Exception('%s is not a dir' % noise_file)
            
            if self.mode == 'all':

                # self.train_data = train_data
                self.noise_label = noise_label
                self.train_indices = np.arange(len(train_label))

            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]

                    self.probability = [probability[i] for i in pred_idx]   
                    clean = (np.array(noise_label)==np.array(train_label)) 
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value() 
                    print('\n'+'Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    log.flush()      
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                
                self.noise_label = [noise_label[i] for i in pred_idx]   
                self.train_indices = pred_idx
                print("%s data has a size of %d"%(self.mode,len(pred_idx)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            # img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            # img = Image.fromarray(img)
            # img1 = self.transform(img) 
            # img2 = self.transform(img) 
            # return img1, img2, target, prob 

            prob = self.probability[index]
            target = self.noise_label[index]
            img1,_ = self.train_dataset.__getitem__(self.train_indices[index])
            img2,_ = self.train_dataset.__getitem__(self.train_indices[index])
            return img1, img2, target, prob         
        
        
        
        elif self.mode=='unlabeled':
            # img = self.train_data[index]
            # img = Image.fromarray(img)
            # img1 = self.transform(img) 
            # img2 = self.transform(img) 
            # return img1, img2
        
            img1,_ = self.train_dataset.__getitem__(self.train_indices[index])
            img2,_ = self.train_dataset.__getitem__(self.train_indices[index])
            return img1, img2
        
        elif self.mode=='all':
            # img, target = self.train_data[index], self.noise_label[index]
            # img = Image.fromarray(img)
            # img = self.transform(img)            
            # return img, target, index 

            target = self.noise_label[index]
            img,_ = self.train_dataset.__getitem__(index)
            return img, target, index
        
        elif self.mode=='test':
            # img, target = self.test_data[index], self.test_label[index]
            # img = Image.fromarray(img)
            # img = self.transform(img)            
            # return img, target
            img, target = self.test_dataset.__getitem__(index)
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            # return len(self.train_data)
            return len(self.train_indices)
        
        else:
            # return len(self.test_data)   
            return self.test_dataset.__len__()
        
        
class cub_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file

        # if self.dataset=='cifar10':
        #     self.transform_train = transforms.Compose([
        #             transforms.RandomCrop(32, padding=4),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        #         ]) 
        #     self.transform_test = transforms.Compose([
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        #         ])    
        # elif self.dataset=='cifar100':    
        #     self.transform_train = transforms.Compose([
        #             transforms.RandomCrop(32, padding=4),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        #         ]) 
        #     self.transform_test = transforms.Compose([
        #             transforms.ToTensor(),
        #             transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        #         ])   

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            
    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = cub_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        
        # elif mode=='warmup_lowP':
        #     all_dataset = cub_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all_lowP",noise_file=self.noise_file)                
        #     trainloader = DataLoader(
        #         dataset=all_dataset, 
        #         batch_size=self.batch_size*2,
        #         shuffle=True,
        #         num_workers=self.num_workers)             
        #     return trainloader
                                       
        elif mode=='train':
            labeled_dataset = cub_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cub_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred)                    
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cub_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cub_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader        