import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from model import CRNN
import editdistance

# data yukleme
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

class IAMDataset(Dataset):
    def __init__(self, indices,train=True):
        self.indices = indices
        self.train=train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = torch.tensor(X[idx], dtype=torch.float32)

        if self.train:
          # gurultu ekleme
          img = img + torch.randn_like(img) * 0.08  
          
          if np.random.rand() < 0.4:
              #parlaklik artirma/azaltma
              img = img * np.random.uniform(0.8, 1.2)
          
          # yazi biraz bulanıklastirma
          if np.random.rand() < 0.3:
              kernel = torch.ones(1, 1, 1, 3) / 3
              img = img.unsqueeze(0)
              img = torch.nn.functional.conv2d(img, kernel, padding=(0,1))
              img = img.squeeze(0)
          
          img = torch.clamp(img, -1, 1)

          if np.random.rand()<0.3:
            img = img * np.random.uniform(0.85, 1.15)
        
        label = torch.tensor(y[idx], dtype=torch.long)
        input_len = torch.tensor(input_lengths[idx], dtype=torch.long)
        label_len = torch.tensor(label_lengths[idx], dtype=torch.long)
        return img, label, input_len, label_len


def collate_fn(batch):
    imgs, labels, in_lens, lbl_lens = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.stack(in_lens), torch.stack(lbl_lens)

n = len(X)
n_val = int(n * 0.2)
all_idx = list(range(n))
np.random.seed(42)
np.random.shuffle(all_idx)
train_idx, val_idx = all_idx[:n - n_val], all_idx[n - n_val:]

train_loader = DataLoader(IAMDataset(train_idx,train=True), batch_size=32, shuffle=True,  collate_fn=collate_fn, num_workers=2)
val_loader   = DataLoader(IAMDataset(val_idx,train=False),   batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=2)

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CRNN(NUM_CLASS).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",patience=8,factor=0.5, min_lr=1e-6)

ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

# greedy decode
def decode(output):
    pred = output.argmax(2).permute(1, 0)
    results = []
    for seq in pred:
        chars, prev = [], None
        for idx in seq.tolist():
            if idx != BLANK_IDX and idx != prev:
                chars.append(idx2char.get(idx, "?"))
            prev = idx
        results.append("".join(chars))
    return results

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0

    total_edit_distance = 0
    total_characters = 0

    with torch.set_grad_enabled(train):
        for imgs, labels, in_lens, lbl_lens in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            out  = model(imgs)
            loss = ctc_loss(out, labels, in_lens, lbl_lens)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            total_loss += loss.item()

            preds = decode(out)
            offset = 0

            for i, length in enumerate(lbl_lens.tolist()):
                gt = "".join([idx2char.get(l, "?") 
                              for l in labels[offset:offset+length].tolist()])

                dist = editdistance.eval(preds[i], gt)

                total_edit_distance += dist
                total_characters += len(gt)

                offset += length

    cer = total_edit_distance / max(total_characters, 1)

    return total_loss / len(loader), cer
    
# egitim
EPOCHS = 60
SAVE_EVERY = 5
EARLY_STOP_PAT = 15

best_val_loss = float("inf")
epochs_no_improve = 0

print("Egitim basladi...\n")

for epoch in range(1, EPOCHS + 1):
    train_loss, train_cer = run_epoch(train_loader, train=True)
    val_loss,   val_cer   = run_epoch(val_loader, train=False)

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} CER: {train_cer:.4f} | "
          f"Val Loss: {val_loss:.4f} CER: {val_cer:.4f}")

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_cer": val_cer}, "best_model.pth")
        print(f"  → Best model kaydedildi (val_loss={val_loss:.4f}, val_cer={val_cer:.1f}%)")
    else:
        epochs_no_improve += 1
        print(f"  Iyilesme yok ({epochs_no_improve}/{EARLY_STOP_PAT})")

    if epoch % SAVE_EVERY == 0:
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer": optimizer.state_dict()}, f"checkpoint_epoch_{epoch}.pth")
        print(f"  → Checkpoint kaydedildi: checkpoint_epoch_{epoch}.pth")

    if epochs_no_improve >= EARLY_STOP_PAT:
        print(f"\nEARLY STOPPING — En iyi val_loss: {best_val_loss:.4f}")
        break

torch.save(model.state_dict(), "iam_word_final_model.pth")
print("\nEgitim tamamlandi.")