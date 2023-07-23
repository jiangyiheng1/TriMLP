import torch
import numpy as np
from collections import Counter


def count(logits, trg, cnt):
    output = logits.clone()
    for i in range(trg.size(0)):
        output[i][0] = logits[i][trg[i]]
        output[i][trg[i]] = logits[i][0]
    idx = output.sort(descending=True, dim=-1)[1]
    order = idx.topk(k=1, dim=-1, largest=False)[1]
    cnt.update(order.squeeze().tolist())
    return cnt


def calculate(cnt, array):
    for k, v in cnt.items():
        array[k] = v
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, len(array)) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr, ndcg


def eval(data_loader, n_item, model, device):
    cnt = Counter()
    array = np.zeros(n_item)
    model.eval()
    with torch.no_grad():
        for _, (src_items, trg_items, data_size) in enumerate(data_loader):
            src = src_items.to(device)
            target = trg_items.to(device)
            data_size = data_size.to(device)
            logits = model(src, data_size)
            cnt = count(logits, target, cnt)
    hr, ndcg = calculate(cnt, array)
    metrics = {'HR/NDCG@1': '{:.5f}'.format(hr[0]),
               'HR@5': '{:.5f}'.format(hr[4]),
               'NDCG@5': '{:.5f}'.format(ndcg[4]),
               'HR@10': '{:.5f}'.format(hr[9]),
               'NDCG@10': '{:.5f}'.format(ndcg[9])}
    for key, value in metrics.items():
        print(key, '=', value)
    return metrics
