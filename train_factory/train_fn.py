import torch
import random
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler
from collate_fn import gen_train_batch, gen_eval_batch
from trainer import trainer
from evaluator import eval
from torch.utils.data import DataLoader


def train(model, max_len, optimizer, n_epoch, train_data, train_bsz, eval_data, eval_bsz, result_pth, save_pth, device):
    reset_random_seed(42)
    eval_loader = DataLoader(dataset=eval_data,
                             batch_size=eval_bsz,
                             num_workers=12,
                             prefetch_factor=2,
                             collate_fn=lambda e: gen_eval_batch(e, eval_data, max_len))
    best_metric = eval(eval_loader, eval_data.n_item, model, device)
    count = 0
    for epoch in range(1, n_epoch, 1):
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=train_bsz,
                                  sampler=LadderSampler(train_data, train_bsz),
                                  num_workers=12,
                                  prefetch_factor=2,
                                  collate_fn=lambda e: gen_train_batch(e, train_data, max_len))
        optimizer = trainer(epoch, train_loader, model, optimizer, device)
        current_metric = eval(eval_loader, eval_data.n_item, model, device)

        indicator = 0
        if current_metric['HR@10'] >= best_metric['HR@10']:
            indicator = 1
        if indicator == 1:
            best_metric = current_metric
            f = open(result_pth, 'w')
            print('Epoch:', epoch, file=f)
            for k, v in best_metric.items():
                print(k, ':', v, file=f)
            f.close()
            best_model = deepcopy(model)
            best_model.save(save_pth)
            count = 0
        else:
            count = count + 1
            if count == 9:
                print('Early Stop!')
                break
    for k, v in best_metric.items():
        print(k, ':', v)


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_size, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_size * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)