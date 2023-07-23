import torch


def pad_sequence(seq, max_len):
    seq = list(seq)
    if len(seq) < max_len:
        seq = seq + [0] * (max_len - len(seq))
    else:
        seq = seq[-max_len:]
    return torch.tensor(seq)


def gen_train_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    for e in src_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size


def gen_eval_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    items, data_size = [], []
    for e in src_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, max_len))
        data_size.append(len(_))
    src_items = torch.stack(items)
    data_size = torch.tensor(data_size)
    items = []
    for e in trg_seq:
        _, i_, _ = zip(*e)
        items.append(pad_sequence(i_, 1))
    trg_items = torch.stack(items)
    return src_items, trg_items, data_size
