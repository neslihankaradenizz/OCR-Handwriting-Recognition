import json
import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR   = "iam_words"
WORDS_TXT  = os.path.join(ROOT_DIR, "words.txt")
IMG_HEIGHT = 32
MAX_WIDTH  = 256

def read_words_file(words_txt_path):
    samples = []
    with open(words_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2 or parts[1] != "ok":
                continue
            samples.append((parts[0], parts[-1]))
    return samples

def get_image_path(root_dir, image_id):
    p1 = image_id[:3]
    p2 = image_id[:7]
    return os.path.join(root_dir, "words", p1, p2, image_id + ".png")

def preprocess_image(img_path, img_height=IMG_HEIGHT, max_width=MAX_WIDTH):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w  = img.shape
    new_w = int(w * (img_height / h))
    new_w = min(new_w, max_width)

    img = cv2.resize(img, (new_w, img_height))
    img = cv2.equalizeHist(img)
    
    # arka plan siyah, yazi beyaz
    img = 255 - img

    # padding siyah (0) ile yap
    padded = np.zeros((img_height, max_width), dtype=np.float32)
    padded[:, :new_w] = img

    padded = padded / 255.0          # 0-1
    padded = (padded - 0.5) / 0.5   # -1 ile 1

    padded = np.expand_dims(padded, axis=0)
    
    return padded.astype(np.float32), new_w

def build_vocab(samples):
    chars = set(" ")
    for _, label in samples:
        chars.update(label)
    char_list = sorted(list(chars))
    
    char2idx = {c: i+1 for i, c in enumerate(char_list)}
    idx2char = {0: "<BLANK>"} 
    idx2char.update({i: c for c, i in char2idx.items()})

    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump({
            "char2idx": char2idx,
            "idx2char": {str(k): v for k, v in idx2char.items()}
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Vocab kaydedildi: {len(char2idx)} karakter + blank")

    return char2idx, idx2char

def encode_label(text, char2idx):
    return [char2idx[c] for c in text if c in char2idx]

def load_iam_dataset(root_dir, words_txt_path, char2idx):
    samples = read_words_file(words_txt_path)
    data, labels, input_lengths, label_lengths = [], [], [], []

    for image_id, label in tqdm(samples, desc="Yukleniyor"):
        img_path = get_image_path(root_dir, image_id)
        if not os.path.exists(img_path):
            continue
        img, new_w = preprocess_image(img_path)
        if img is None:
            continue
        encoded = encode_label(label, char2idx)
        if len(encoded) == 0:
            continue

        # CNN yatayda 4'e boler (MaxPool 2,2 x1 + MaxPool 2,1 x2)
        seq_len = new_w // 4
        if seq_len < len(encoded):     
            continue

        data.append(img)
        labels.append(encoded)
        input_lengths.append(seq_len)
        label_lengths.append(len(encoded))

    return np.array(data), labels, np.array(input_lengths), np.array(label_lengths)

if __name__ == "__main__":
    samples = read_words_file(WORDS_TXT)
    print(f"Toplam örnek: {len(samples)}")

    char2idx, idx2char = build_vocab(samples)
    print(f"Vocab boyutu (blank hariç): {len(char2idx)}")

    X, y, input_len, label_len = load_iam_dataset(ROOT_DIR, WORDS_TXT, char2idx)
    print(f"X shape     : {X.shape}")
    print(f"Örnek label : {y[0]}")

    np.save("X.npy", X)
    np.save("input_lengths.npy", input_len)
    np.save("label_lengths.npy", label_len)

    with open("labels.json", "w") as f:
        json.dump(y, f)
