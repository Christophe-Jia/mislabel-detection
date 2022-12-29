import torch.nn as nn

class LSTM(nn.Module):
    
    def __init__(self,in_dim,hidden_dim=64,n_layer=2,n_class=2,dropout=0):
        super(LSTM,self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim,n_class)
        
        # initialisation
        self.lstm = nn.LSTM(in_dim,hidden_dim,n_layer,batch_first=True,dropout=self.dropout, bidirectional=False)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): 
                nn.init.orthogonal_(param)
            else:
                param.data.fill_(0.01)
        
    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.classifier(out)
        return out
    




