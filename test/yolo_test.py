import torch, cv2

# I have loaded yolov5 and pretrained
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# cam opening
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame[..., ::-1])  # BGR → RGB
    annotated = results.render()[0][..., ::-1]  # back to BGR
    cv2.imshow("YOLOv5 Webcam", annotated)
    if cv2.waitKey(1) == 27:  
        break

cap.release()
cv2.destroyAllWindows()
