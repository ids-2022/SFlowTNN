import copy
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
class MyTextCnn(nn.Module):
    def __init__(self, dim, n_filter, filter_size, out_dim):

        super(MyTextCnn,self).__init__()

        self.cov=nn.ModuleList([
            nn.Conv1d(in_channels=1,
                      out_channels=n_filter,
                      stride=(fs,),
                      kernel_size=(fs,dim)) for fs in filter_size

        ])
        self.cov2 = nn.ModuleList([
            nn.Conv1d(in_channels=n_filter,
                      out_channels=2*n_filter,
                      stride=(fs,),
                      kernel_size=(fs,1)) for fs in filter_size

        ])
        self.fc=nn.Linear(len(filter_size)*(2*n_filter)*2,out_dim)

    def k_max_pool(self,x,dim,k):
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self,x):

        #coved = [torch.relu(cov(x)).squeeze(3) for cov in self.cov]
        coved = [torch.relu(cov(x)) for cov in self.cov]

        coved_twice=[]
        for i,x in enumerate(coved):
            coved_twice.append(torch.relu(self.cov2[i](x)))

        pooled=[self.k_max_pool(cov,dim=2,k=2) for cov in coved_twice]

        #cat = self.dropout(torch.cat(pooled,dim=1))
        cat = torch.cat(pooled,dim=1)
        cat = torch.flatten(cat, 1)

        return self.fc(cat)