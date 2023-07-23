import os
import copy
import math
import numpy as np
from torch.utils.data import Dataset
from utils import serialize, un_serialize


class InterData(Dataset):
    def __init__(self, path, cold_item, cold_user):
        super(InterData, self).__init__()
        self.item2idx = {'<pad>': 0}
        self.idx2item = {0: '<pad>'}
        self.item2count = {}
        self.n_item = 1
        self.build_vocab(path, cold_item)
        self.n_user, self.user2idx, self.user_seq = self.process(path, cold_user)

    def build_vocab(self, path, cold_item):
        for line in open(path):
            line = line.strip().split('\t')
            if line[0] == 'user_id:token':
                continue
            item = line[1]
            self.add_item(item)
        if cold_item > 0:
            self.n_item = 1
            self.item2idx = {'<pad>': 0}
            self.idx2item = {0: '<pad>'}
            for item in self.item2count:
                if self.item2count[item] >= cold_item:
                    self.add_item(item)

    def add_item(self, item):
        if item not in self.item2idx:
            self.item2idx[item] = self.n_item
            self.idx2item[self.n_item] = item
            if item not in self.item2count:
                self.item2count[item] = 1
            self.n_item += 1
        else:
            self.item2count[item] += 1

    def process(self, path, cold_user):
        n_user = 1
        user2idx = {}
        user_seq = {}
        user_seq_array = list()
        for line in open(path):
            line = line.strip().split('\t')
            if line[0] == 'user_id:token':
                continue
            if len(line) >= 3:
                user, item, timestamp = line[0], line[1], float(line[2])
            else:
                user, item, timestamp = line[0], line[1], 0
            if item not in self.item2idx:
                continue
            item_idx = self.item2idx[item]
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([item_idx, timestamp])

        for user, seq in user_seq.items():
            if len(seq) >= cold_user:
                user2idx[user] = n_user
                user_idx = n_user
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for item_idx, _ in sorted(seq, key=lambda e: e[-1]):
                    if item_idx in tmp_set:
                        seq_new.append((user_idx, item_idx, True))
                    else:
                        seq_new.append((user_idx, item_idx, False))
                        tmp_set.add(item_idx)
                        cnt += 1
                if cnt >= (cold_user / 2):
                    n_user += 1
                    user_seq_array.append(seq_new)
        return n_user, user2idx, user_seq_array

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def partition(self, max_len):
        train_data = copy.copy(self)
        eval_data = copy.copy(self)
        train_seq = []
        eval_seq = []
        for user in range(len(self)):
            seq = self[user]
            for i in reversed(range(len(seq))):
                if not seq[i][-1]:
                    break
            trg = seq[i: i+1]
            src = seq[max(0, i-max_len): i]
            eval_seq.append((src, trg))

            n_sample = math.floor((i+max_len-1)/max_len)
            for k in range(n_sample):
                if (i-k*max_len) > max_len*1.1:
                    trg = seq[i-(k+1)*max_len: i-k*max_len]
                    src = seq[i-(k+1)*max_len-1: i-k*max_len-1]
                    train_seq.append((src, trg))
                else:
                    trg = seq[1: i-k*max_len]
                    src = seq[0: i-k*max_len-1]
                    train_seq.append((src, trg))
                    break
        train_data.user_seq = train_seq
        eval_data.user_seq = eval_seq
        return train_data, eval_data


raw_data_pth = '' + '.txt'
clean_data_pth = '' + 'data'
clean_data_stat = '' + '.txt'

if os.path.exists(clean_data_pth):
    dataset = un_serialize(clean_data_pth)
else:
    cold_item_threshold = 10
    cold_user_threshold = 20
    dataset = InterData(raw_data_pth, cold_item_threshold, cold_user_threshold)
count = 0
length = []
for seq in dataset.user_seq:
    count += len(seq)
    length.append(len(seq))
print("#Users:", dataset.n_user - 1)
print("#Items:", dataset.n_item - 1)
print("#Interactions:", count)
print("Average Sequence Length:", np.mean(np.array(length)))
print("Sparsity:", 1 - count / ((dataset.n_user - 1) * (dataset.n_item - 1)))
serialize(dataset, clean_data_pth)

f = open(clean_data_stat, 'w')
print("#Users:", dataset.n_user - 1, file=f)
print("#Items:", dataset.n_item - 1, file=f)
print("#Interactions:", count, file=f)
print("Average Sequence Length:", np.mean(np.array(length)), file=f)
print("Sparsity:", 1 - count / ((dataset.n_user - 1) * (dataset.n_item - 1)), file=f)
f.close()

