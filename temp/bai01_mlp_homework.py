"""Bài 1 - BTVN: khảo sát hidden size và learning rate trên make_moons."""
import itertools, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_moons
from common import seed_everything, train_classifier

seed_everything(42)
X,y=make_moons(n_samples=1200,noise=0.28,random_state=42)
ds=TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long))
tr,va=random_split(ds,[960,240],generator=torch.Generator().manual_seed(42))

for hidden,lr in itertools.product([8,32,64],[1e-2,1e-3]):
    print('\nCONFIG',hidden,lr)
    model=nn.Sequential(nn.Linear(2,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))
    train_classifier(model,DataLoader(tr,32,shuffle=True),DataLoader(va,256),epochs=15,lr=lr)
