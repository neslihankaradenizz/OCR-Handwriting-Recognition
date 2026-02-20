import torch
import editdistance
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import CRNN

# data
X = np.load("X.npy")
input_lengths = np.load("input_lengths.npy")
label_lengths = np.load("label_lengths.npy")
with open("labels.json") as f:
    y = json.load(f)
with open("vocab.json") as f:
    vocab = json.load(f)

char2idx = vocab["char2idx"]
idx2char = {int(k): v for k, v in vocab["idx2char"].items()}
NUM_CLASS = len(char2idx) + 1
BLANK_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
class IAMDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        img = torch.tensor(X[idx], dtype=torch.float32)
        label = torch.tensor(y[idx], dtype=torch.long)
        input_len = torch.tensor(input_lengths[idx], dtype=torch.long)
        label_len = torch.tensor(label_lengths[idx], dtype=torch.long)
        return img, label, input_len, label_len

def collate_fn(batch):
    imgs, labels, in_lens, lbl_lens = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.stack(in_lens), torch.stack(lbl_lens)

# val seti 
n = len(X)
n_val = int(n * 0.1)
all_idx = list(range(n))
np.random.seed(42)
np.random.shuffle(all_idx)
val_idx = all_idx[n - n_val:]

val_loader = DataLoader(IAMDataset(val_idx), batch_size=64,
                        shuffle=False, collate_fn=collate_fn, num_workers=2)

# model
model = CRNN(NUM_CLASS).to(device)
checkpoint = torch.load("/content/sonkezz/checkpoint_epoch_45.pth", map_location=device, weights_only=True) #colab adres
model.load_state_dict(checkpoint["model_state"])
model.eval()
print("model yuklendi...")

# decode
def decode(output):
    pred = output.argmax(2).permute(1, 0)
    results = []
    for seq in pred:
        chars, prev = [], None
        for idx in seq.tolist():
            if idx != BLANK_IDX and idx != prev:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results

# CER ve WER hesabi
def cer(gt, pred):
    return editdistance.eval(gt, pred) / max(len(gt), 1)

def wer(gt, pred):
    gt_words   = gt.split()
    pred_words = pred.split()
    return editdistance.eval(gt_words, pred_words) / max(len(gt_words), 1)

all_cer, all_wer = [], []

with torch.no_grad():
    for imgs, labels, in_lens, lbl_lens in val_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        preds = decode(out)

        offset = 0
        for i, length in enumerate(lbl_lens.tolist()):
            gt = "".join([idx2char.get(l, "") for l in labels[offset:offset+length].tolist()])
            pr = preds[i]

            all_cer.append(cer(gt.lower(), pr.lower()))
            all_wer.append(wer(gt.lower(), pr.lower()))
            offset += length

print(f"Avg CER: {sum(all_cer)/len(all_cer)*100:.2f}%")
print(f"Avg WER: {sum(all_wer)/len(all_wer)*100:.2f}%")