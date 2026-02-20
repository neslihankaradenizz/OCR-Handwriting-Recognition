import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segment import segment_all
from inference import predict_word

BG = "#F4F6F8"
CARD = "#FFFFFF"
PRIMARY = "#4F8EF7"
PRIMARY_HOVER = "#3A73D9"
SECONDARY = "#F5B971"
SECONDARY_HOVER = "#E2A45E"
TEXT = "#2B2B2B"
GRAY = "#7A7A7A"
BORDER = "#E6EAF0"

EXAMPLES = ["ornek11.jpeg", "ornek7.jpg", "ornek8.jpg"]


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("El Yazısı Tanıma (OCR)")
        self.root.geometry("1050x720")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.build_ui()

    def build_ui(self):
        # HEADER
        header = tk.Frame(self.root, bg=PRIMARY, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text=" El Yazısı Tanıma Sistemi",
            font=("Segoe UI", 20, "bold"),
            bg=PRIMARY,
            fg="white"
        ).pack(side="left", padx=30, pady=18)

        # CONTENT
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=30, pady=20)

        left = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 20))

        right = tk.Frame(content, bg=BG, width=300)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        # IMAGE CARD
        image_card = tk.Frame(left, bg=CARD, highlightbackground=BORDER,
                            highlightthickness=1)
        image_card.pack(fill="both", expand=True, pady=(0, 15))

        tk.Label(
            image_card,
            text="Görüntü",
            font=("Segoe UI", 13, "bold"),
            bg=CARD,
            fg=TEXT
        ).pack(anchor="w", padx=15, pady=(15, 5))

        self.img_label = tk.Label(
            image_card,
            text="Tıklayarak görüntü yükleyin",
            font=("Segoe UI", 12),
            bg="#F9FAFB",
            fg=GRAY,
            height=15,
            cursor="hand2"
        )
        self.img_label.pack(fill="both", expand=True, padx=15, pady=10)
        self.img_label.bind("<Button-1>", lambda e: self.upload_image())


        # BUTTON ROW
        btn_frame = tk.Frame(left, bg=BG)
        btn_frame.pack(fill="x", pady=10)

        self.upload_btn = self.create_button(btn_frame, "📂 Görüntü Seç",
                                             PRIMARY, PRIMARY_HOVER,
                                             self.upload_image)
        self.upload_btn.pack(side="left", padx=5)

        self.run_btn = self.create_button(btn_frame, "▶ Tanı",
                                          SECONDARY, SECONDARY_HOVER,
                                          self.run_ocr)
        self.run_btn.pack(side="left", padx=5)

        self.clear_btn = self.create_button(btn_frame, "✕ Temizle",
                                            "#D9534F", "#C9302C",
                                            self.clear)
        self.clear_btn.pack(side="left", padx=5)

        # RESULT CARD
        result_card = tk.Frame(left, bg=CARD, highlightbackground=BORDER,
                               highlightthickness=1)
        result_card.pack(fill="both", expand=True, pady=(10, 0))

        tk.Label(
            result_card,
            text="Tanınan Metin",
            font=("Segoe UI", 12, "bold"),
            bg=CARD,
            fg=TEXT
        ).pack(anchor="w", padx=15, pady=(15, 5))

        self.result_text = tk.Text(
            result_card,
            font=("Segoe UI", 12),
            bg="#F9FAFB",
            fg=TEXT,
            relief="flat",
            wrap="word",
            height=6
        )
        self.result_text.pack(fill="both", expand=True, padx=15, pady=10)

        # STATUS
        self.status_label = tk.Label(
            left,
            text="Hazır",
            font=("Segoe UI", 9),
            bg=BG,
            fg=GRAY
        )
        self.status_label.pack(anchor="w", pady=6)

        # EXAMPLES PANEL
        tk.Label(
            right,
            text="Örnek Görseller",
            font=("Segoe UI", 12, "bold"),
            bg=BG,
            fg=TEXT
        ).pack(anchor="w", pady=(0, 10))

        for path in EXAMPLES:
            if os.path.exists(path):
                self.add_example_card(right, path)

    def create_button(self, parent, text, color, hover, command):
        btn = tk.Button(
            parent,
            text=text,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg="white",
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
            command=command
        )

        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    def add_example_card(self, parent, path):
        card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="x", pady=6)

        img = Image.open(path)
        img.thumbnail((260, 80))
        photo = ImageTk.PhotoImage(img)

        lbl = tk.Label(card, image=photo, bg=CARD, cursor="hand2")
        lbl.image = photo
        lbl.pack(padx=10, pady=8)
        lbl.bind("<Button-1>", lambda e, p=path: self.load_image(p))

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.load_image(path)

    def load_image(self, path):
        self.uploaded_path = path
        img = Image.open(path)
        img.thumbnail((500, 250))
        photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=photo, text="")
        self.img_label.image = photo
        self.status_label.config(text=f"Yüklendi: {os.path.basename(path)}")

    def run_ocr(self):
        if not hasattr(self, "uploaded_path"):
            self.status_label.config(text="Önce görüntü yükleyin!")
            return

        self.status_label.config(text="İşleniyor...")
        self.root.update()

        try:
            words, bboxes, img = segment_all(self.uploaded_path)
            predictions = []
            for w in words:
                pred = predict_word(w)
                if pred:
                    predictions.append(pred)

            result = " ".join(predictions) if predictions else "Metin bulunamadı."
            self.status_label.config(text=f"{len(predictions)} kelime tanındı.")

        except Exception as e:
            result = f"Hata: {str(e)}"
            self.status_label.config(text="Bir hata oluştu.")

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", result)

    def clear(self):
        self.img_label.configure(image="", text="Tıklayarak görüntü yükleyin")
        self.img_label.image = None
        self.result_text.delete("1.0", "end")
        self.status_label.config(text="Hazır")
        if hasattr(self, "uploaded_path"):
            del self.uploaded_path

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()