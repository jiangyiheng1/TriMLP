import torch
from model.trimlp import TriMLP
from train_factory.train_fn import train
from data_factory.utils import un_serialize
from data_factory.process import InterData

d_model = 128
dropout = 0.5
device = 'cuda:0'
n_epoch = 1000
train_bsz = 128
eval_bsz = 64

data_pth = '' + '.data'
dataset = un_serialize(data_pth)
n_item = dataset.n_item
max_len = 32
n_session = 2
train_data, eval_data = dataset.partition(max_len)

result_pth = '' + '.txt'
save_pth = '' + '.pkl'
model = TriMLP(n_item, d_model, max_len, n_session, dropout)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(model, max_len, optimizer, n_epoch, train_data, train_bsz, eval_data, eval_bsz, result_pth, save_pth, device)
