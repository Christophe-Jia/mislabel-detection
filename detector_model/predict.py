import torch
from torch.autograd import Variable
import torch.nn.functional as F

import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

"""
Returns a ranking of indexes, from small to large according to the probability of being predicted as noisy label
"""
def get_order(loader,noise_detector):    
    noise_detector.eval()
    predictions=[]
    labels=[]
    for data in tqdm.tqdm(loader,desc="Ranking all training samples"):
        img, label = data
        img = img.float()

        with torch.no_grad():
            img = Variable(img).cuda()
      
        out = noise_detector(img)
        out = F.softmax(out)
        out = out.cpu().detach().numpy()

        predictions.extend(out[:,1])
        labels.extend(label)

    pred=[1 if x>=0.5 else 0 for x in predictions]
    TP=sum([1 for i in range(len(pred)) if (pred[i]==1 and labels[i]==1)])
    TN=sum([1 for i in range(len(pred)) if (pred[i]==0 and labels[i]==0)])
    FN=sum([1 for i in range(len(pred)) if (pred[i]==0 and labels[i]==1)])
    FP=sum([1 for i in range(len(pred)) if (pred[i]==1 and labels[i]==0)])
    
    try:
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('\n With a threshold = 0.5, we make a binary classification with Precision: {:.6f}, Recall: {:.6f}, F1: {:.6f}, Acc: {:.6f}'.format(p,r,F1,acc))
        
        auc = roc_auc_score(labels, predictions)
        print(auc)
        map = average_precision_score(labels, predictions)
        print(map)
    
    except:
        auc=map=0

    order = np.argsort(predictions)
    return order,auc,map