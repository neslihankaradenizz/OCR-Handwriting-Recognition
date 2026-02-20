import torch
import cv2
import re
import numpy as np
import json
from model import CRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("checkpoint_epoch_45.pth", map_location=device, weights_only=True)

with open("vocab.json") as f: vocab = json.load(f)
char2idx = vocab["char2idx"]
idx2char = {int(k): v for k, v in vocab["idx2char"].items()}

model = CRNN(len(char2idx) + 1).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print(f"Model yüklendi...")

def preprocess_word(word_img, img_height=32, max_width=256):
    h, w = word_img.shape
    new_w = int(w * (img_height / h))
    new_w = min(new_w, max_width)

    img = cv2.resize(word_img, (new_w, img_height))
    img = cv2.equalizeHist(img)
    img = 255 - img  

    padded = np.zeros((img_height, max_width), dtype=np.float32)
    padded[:, :new_w] = img
    padded = padded / 255.0
    padded = (padded - 0.5) / 0.5

    inp = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return inp.to(device)

def predict_word(word_img):
    inp = preprocess_word(word_img) 

#greedy decode 
def decode(output):
    pred = output.argmax(2).squeeze(1)
    chars, prev = [], None
    for idx in pred.tolist():
        if idx != 0 and idx != prev:
            chars.append(idx2char.get(idx, ""))
        prev = idx
    result = "".join(chars)
    
    result = re.sub(r'[^\w\s\-\']', '', result)  
    result = result.strip()
    return result

def predict_word(word_img):
    inp = preprocess_word(word_img)
    with torch.no_grad():
        out = model(inp)
        return decode(out)