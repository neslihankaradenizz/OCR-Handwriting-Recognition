# El Yazısı Tanıma Sistemi (OCR Handwriting Recognition)

Derin öğrenme tabanlı bir el yazısı tanıma (OCR) sistemi. CRNN (Convolutional Recurrent Neural Network) mimarisi ve CTC (Connectionist Temporal Classification) kaybı kullanılarak IAM el yazısı veri seti üzerinde eğitilmiştir.

---

## Özellikler

- **CRNN Mimarisi**: CNN + BiLSTM hibrit ağ yapısı ile karakter dizisi tanıma
- **CTC Kaybı**: Hizalama gerektirmeyen değişken uzunluklu dizi eğitimi
- **Kelime Segmentasyonu**: Görüntüden otomatik kelime bölgeleri çıkarımı
- **Masaüstü GUI**: Tkinter tabanlı kullanıcı arayüzü
- **Değerlendirme Metrikleri**: CER (Karakter Hata Oranı) ve WER (Kelime Hata Oranı)

---

## Proje Yapısı

```
OCR-Handwriting-Recognition/
├── app.py                # Tkinter GUI uygulaması
├── model.py              # CRNN model mimarisi
├── train.py              # Model eğitim scripti
├── inference.py          # Tahmin ve çözümleme fonksiyonları
├── preprocess.py         # Veri seti ön işleme pipeline'ı
├── segment.py            # Görüntüden kelime segmentasyonu
├── segment_pipeline.py   # Tam OCR pipeline orkestratörü
├── hata.py               # CER/WER değerlendirme scripti
├── labels.json           # Eğitim verisi etiketleri
└── vocab.json            # Karakter-indeks eşleme sözlüğü
```

---

## Kurulum

### Gereksinimler

```bash
pip install torch torchvision opencv-python numpy pillow tqdm editdistance
```

### Bağımlılıklar

| Kütüphane | Kullanım Amacı |
|-----------|---------------|
| PyTorch | Model tanımlama ve eğitim |
| OpenCV | Görüntü işleme |
| NumPy | Dizi operasyonları |
| Tkinter | Masaüstü GUI (Python ile birlikte gelir) |
| Pillow | GUI görüntü yönetimi |
| editdistance | CER/WER hesaplama |
| tqdm | İlerleme çubukları |

---

## Kullanım

### 1. Veri Seti Hazırlama

IAM veri setini aşağıdaki yapıda `iam_words/` dizinine yerleştirin:

```
iam_words/
├── words.txt           # Meta veri dosyası
└── words/
    └── <partition>/
        └── <form>/
            └── <word_id>.png
```

Ardından ön işleme scriptini çalıştırın:

```bash
python preprocess.py
```

Bu komut `X.npy`, `input_lengths.npy`, `label_lengths.npy`, `labels.json` ve `vocab.json` dosyalarını oluşturur.

### 2. Model Eğitimi

```bash
python train.py
```

- 60 epoch, erken durdurma (15 epoch iyileşme olmazsa)
- Adam optimizer (lr=3e-4), ReduceLROnPlateau scheduler
- En iyi model `best_model.pth` olarak kaydedilir

### 3. Modeli Değerlendirme

```bash
python hata.py
```

Doğrulama seti üzerinde CER ve WER metriklerini hesaplar.

### 4. GUI Uygulamasını Çalıştırma

```bash
python app.py
```

Görüntü yükleyerek el yazısını metin olarak çıktı alabilirsiniz.

### 5. Tam Pipeline

```bash
python segment_pipeline.py
```

Bir görüntü üzerinde kelime segmentasyonu ve OCR'ı birlikte çalıştırır.

---

## Model Mimarisi

```
Giriş Görüntüsü (1 × 32 × 256)
        │
┌───────▼────────────────────────────────────────┐
│  CNN Bloğu (5× Conv → BN → ReLU → MaxPool)     │
│  Kanallar: 1 → 64 → 128 → 256 → 512            │
│  Çıkış: (B, 512, 1, W/4)                       │
└───────────────────────────────┬────────────────┘
                                │ Reshape
┌───────────────────────────────▼────────────────┐
│  BiLSTM (2 katman, hidden=256, dropout=0.5)     │
│  Çıkış: (B, W/4, 512)                          │
└───────────────────────────────┬────────────────┘
                                │
┌───────────────────────────────▼────────────────┐
│  Fully Connected + Log-Softmax                 │
│  Çıkış: (B, W/4, num_classes)                  │
└────────────────────────────────────────────────┘
        │
   CTC Decode
        │
   Tahmin Metni
```

---

## Eğitim Detayları

| Parametre | Değer |
|-----------|-------|
| Batch boyutu | 32 (eğitim), 64 (doğrulama) |
| Epoch sayısı | 60 (erken durdurma ile) |
| Optimizer | Adam (lr=3e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=8) |
| Dropout | 0.3 – 0.6 (CNN), 0.5 (LSTM) |
| Veri artırma | Gaussian gürültü, parlaklık, bulanıklık |
| Train/Val oranı | 80% / 20% |

---

## Kelime Segmentasyon Pipeline'ı

1. **Binarizasyon**: Adaptif eşikleme
2. **Morfolojik genişleme**: Bileşenleri birleştirme
3. **Kontur tespiti**: Sınırlayıcı kutu çıkarımı
4. **Filtreleme**: Boyut ve en-boy oranına göre
5. **Sıralama**: Soldan sağa x-koordinatına göre

---

## Veri Seti

**IAM El Yazısı Veri Seti** — İngilizce el yazısı kelimeler

- Görüntü boyutu: 32 × 256 piksel (normalize edilmiş)
- Sözlük büyüklüğü: 80 karakter (harf, rakam, noktalama, boşluk)
- Görüntü ön işleme: histogram eşitleme, ters çevirme, [-1, 1] normalizasyonu

---

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.
