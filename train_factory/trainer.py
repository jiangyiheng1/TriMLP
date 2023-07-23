import time
from tqdm import tqdm
import torch.nn.functional as F


def trainer(epoch, data_loader, model, optimizer, device):
    print('+' * 30, 'Epoch {}'.format(epoch), '+' * 30)
    start_time = time.time()
    model.train()
    running_loss = 0.0
    processed_batch = 0
    batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
    for batch_idx, (src_items, trg_items, data_size) in batch_iterator:
        optimizer.zero_grad()
        src = src_items.to(device)
        target = trg_items.to(device)
        data_size = data_size.to(device)
        logits = model(src, data_size)
        logits = logits.view(-1, logits.size(-1))
        target = target.view(-1)
        loss = F.cross_entropy(logits, target, ignore_index=0)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
        processed_batch = processed_batch + 1
        batch_iterator.set_postfix_str('Loss={:.4f}'.format(loss.item()))
    cost_time = time.time() - start_time
    avg_loss = running_loss / processed_batch
    print('Time={:.4f}, Average Loss={:.4f}'.format(cost_time, avg_loss))
    return optimizer
