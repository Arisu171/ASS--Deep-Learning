"""Bài 1 - trên lớp: MLP phân loại make_moons."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import make_moons
from common import seed_everything, train_classifier

seed_everything(42)
X,y=make_moons(n_samples=800,noise=0.25,random_state=42)
ds=TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.long))
tr,va=random_split(ds,[640,160],generator=torch.Generator().manual_seed(42))
train_loader=DataLoader(tr,32,shuffle=True); val_loader=DataLoader(va,128)
model=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,2))
print(model)
train_classifier(model,train_loader,val_loader,epochs=20,lr=1e-3)
