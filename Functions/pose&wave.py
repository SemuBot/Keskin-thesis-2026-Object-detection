import cv2
import math
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

# COCO Skeleton Connections (Which point connects to what)
SKELETON_LINES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (11, 12), (5, 11), (6, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
]

camera = cv2.VideoCapture(0)
hand_x_history = [] # For waving detection

while camera.isOpened():
    ret, frame = camera.read()
    if not ret: break

    # Flipping the image (no need for all cameras)
    frame = cv2.flip(frame, 1)

    results = model(frame, verbose=False, conf=0.3)

    for r in results:
        if r.keypoints is None or len(r.keypoints.data) == 0:
            continue

        # Get all 17 keypoints
        points = r.keypoints.data[0].cpu().numpy()
        bbox = r.boxes.xyxy[0].cpu().numpy().astype(int)

        # 1. DRAW SKELETON (Drawing lines between COCO points)
        for start_idx, end_idx in SKELETON_LINES:
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            if pt1[2] > 0.3 and pt2[2] > 0.3: # If both points are visible
                cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 2)

        # 2. DRAW POINTS (Circles on each joint)
        for i in range(17):
            x, y, conf = points[i]
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        # 3. ANALYSIS 
        # shoulders (5,6) and eyes (1,2) better sitting 
        l_sh, r_sh = points[5], points[6]
        l_hip, r_hip = points[11], points[12]
        r_wrist = points[10]

        posture = "Scanning..." 
        
        # If at least shoulders are visible
        if l_sh[2] > 0.3 or r_sh[2] > 0.3:
            # Calculate torso angle
            # If hip is not visible (sitting at desk), we assume vertical
            if l_hip[2] > 0.3 or r_hip[2] > 0.3:
                # Full body analysis
                mid_sh_y = (l_sh[1] + r_sh[1]) / 2
                mid_hip_y = (l_hip[1] + r_hip[1]) / 2
                
                # Check aspect ratio of bounding box
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                
                if w > h * 1.5:
                    posture = "Lying"
                elif h < (frame.shape[0] * 0.7):
                    posture = "Sitting"
                else:
                    posture = "Standing"
            else:
                # If only upper body is visible (Common in my laptop camera)
                posture = "Sitting (Upper Body Only)"

        # 4. WAVING DETECTION
        waving_status = "No"
        if r_wrist[2] > 0.3 and r_sh[2] > 0.3:
            if r_wrist[1] < r_sh[1]: # Hand is above shoulder
                hand_x_history.append(r_wrist[0])
                if len(hand_x_history) > 15: hand_x_history.pop(0)
                
                if len(hand_x_history) > 10:
                    diff = max(hand_x_history) - min(hand_x_history)
                    if diff > 40: waving_status = "YES!"
            else:
                hand_x_history = []

        # 5. FINAL UI
        color = (0, 0, 255) if waving_status == "YES!" else (0, 255, 0)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"Pose: {posture}", (bbox[0], bbox[1]-10), 0, 0.6, color, 2)
        cv2.putText(frame, f"Waving: {waving_status}", (bbox[0], bbox[1]+25), 0, 0.6, (255, 255, 0), 2)

    cv2.imshow("SemuBot Keskin", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

camera.release()
cv2.destroyAllWindows()