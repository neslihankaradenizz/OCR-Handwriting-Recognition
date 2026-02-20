import cv2
from segment import segment_all
from inference import predict_word

def ocr_full_page(image_path, visualize=False):
    words, bboxes, img = segment_all(image_path)
    predictions = []
    
    for w in words:
        pred = predict_word(w)
        if pred and len(pred) > 0:
            predictions.append(pred)

    if visualize:
        vis = img.copy()
        for i, (bbox, pred) in enumerate(zip(bboxes, predictions)):
            x0, y0, x1, y1 = bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(vis, pred, (x0, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Segmentasyon", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return " ".join(predictions)

result = ocr_full_page("ornek7.jpg", visualize=True)
print(result)