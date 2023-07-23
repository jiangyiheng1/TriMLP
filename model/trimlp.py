import torch
import torch.nn as nn
from layers import TriMixer


class TriMLP(nn.Module):
    def __init__(self, n_item, d_model, seq_len, n_session, dropout=0.5, act_fn=nn.Sigmoid()):
        super().__init__()
        self.emb_item = nn.Embedding(n_item, d_model, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.mixer = TriMixer(seq_len, n_session, act_fn)
        self.final_layer = nn.Linear(d_model, n_item)

    def forward(self, x, dsz):
        x = self.drop(self.emb_item(x))
        mixer_output = self.mixer(x)
        if self.training:
            output = self.final_layer(mixer_output)
        else:
            output = mixer_output[torch.arange(dsz.size(0)), dsz - 1, :]
            output = self.final_layer(output).detach()
        return output

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
