import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD = 0
UNK = 1
BOS = 2
EOS = 3

OUT_DIR = "../tokenizer/"
IN_DIR = "../Dataset_A1/"

class CustomDataset(Dataset):
    def __init__(self, src, tgt):
        assert len(src) == len(tgt)

        filtered = [
            (s, t) for s, t in zip(src, tgt)
            if len(s)+2 <= 256 and len(t)+2 <= 256
        ]
        self.src, self.tgt = zip(*filtered) if filtered else ([], [])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, key):
        x = [BOS]+self.src[key]+[EOS]
        y = [BOS]+self.tgt[key]+[EOS]
        
        return x,y

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = [
        torch.tensor(x, dtype=torch.long)
        for x in src_batch
    ]

    tgt_batch = [
        torch.tensor(x, dtype=torch.long)
        for x in tgt_batch
    ]

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD
    )

    return src_batch, tgt_batch

