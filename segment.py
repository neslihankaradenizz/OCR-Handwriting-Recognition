import cv2
import numpy as np

def segment_words(gray):
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15
    )

    # kelime seviyesinde birlestirme
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(th, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    word_images = []
    bboxes = []

    h_img, w_img = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 25 or h < 20:
            continue

        aspect=w/h
        if aspect< 0.1:
            continue

        # ilk kaba crop
        word_gray = gray[y:y+h, x:x+w]
        word_bin  = th[y:y+h, x:x+w]

        ys, xs = np.where(word_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        word_img = word_gray[y0:y1+1, x0:x1+1]

        h2, w2 = word_img.shape
        if w2 < 15 or h2 < 10:
            continue

        # global bbox 
        gx0 = x + x0
        gy0 = y + y0
        gx1 = x + x1
        gy1 = y + y1

        word_images.append(word_img)
        bboxes.append((gx0, gy0, gx1, gy1))

    if len(word_images) == 0:
        return [], []

    # soldan sağa sırala
    sorted_items = sorted(zip(word_images, bboxes), key=lambda x: x[1][0])
    word_images, bboxes = zip(*sorted_items)

    return list(word_images), list(bboxes)

def segment_all(path):
    import numpy as np

    file_bytes = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise Exception("Görüntü okunamadı!")

    #img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    words, bboxes = segment_words(gray)
    return words, bboxes, img