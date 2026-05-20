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

camera = cv2.VideoCapture(1) # change to 1 for the Intel
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Sadece bu kısmı sözlük (dictionary) yapısına çevirdik ki her ID'nin kendi geçmişi olsun
people_history = {} 
fall_alert_timer = 0  

while camera.isOpened():
    ret, frame = camera.read()
    if not ret: break

    # Flipping the image (no need for all cameras)
    frame = cv2.flip(frame, 1)

    # ID 
    results = model.track(frame, verbose=False, conf=0.3, persist=True)

    for r in results:
        # Prevent crashes if no one is in the frame (ID control)
        if r.keypoints is None or r.boxes is None or r.boxes.id is None or len(r.boxes) == 0:
            continue

        # Loop through all detected people in the frame
        for i in range(len(r.boxes)):
            # Kişinin ID numarasını alıyoruz
            person_id = int(r.boxes.id[i])

            # ID
            if person_id not in people_history:
                people_history[person_id] = {'hand_x': [], 'torso_y': []}

            # Get all 17 keypoints for person 'i'
            points = r.keypoints.data[i].cpu().numpy()
            bbox = r.boxes.xyxy[i].cpu().numpy().astype(int)

            # Shoulders (5,6), hips (11,12), and right wrist (10)
            l_sh, r_sh = points[5], points[6]
            l_hip, r_hip = points[11], points[12]
            r_wrist = points[10]

            # CRITICAL FILTER: Ignore false positives (like an isolated hand)
            if l_sh[2] < 0.3 and r_sh[2] < 0.3:
                continue

            # 1. DRAW SKELETON (Drawing lines between COCO points)
            for start_idx, end_idx in SKELETON_LINES:
                pt1 = points[start_idx]
                pt2 = points[end_idx]
                if pt1[2] > 0.3 and pt2[2] > 0.3: # If both points are visible
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 2)

            # 2. DRAW POINTS (Circles on each joint)
            for j in range(17):
                x, y, conf = points[j]
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

            # 3. ULTRA-SIMPLE POSTURE ANALYSIS (Using Torso coordinates)
            posture = "Scanning..." 
            
            # If both shoulders and hips are visible
            if (l_sh[2] > 0.3 or r_sh[2] > 0.3) and (l_hip[2] > 0.3 or r_hip[2] > 0.3):
                
                # Find the center of shoulders and center of hips
                mid_sh_x = (l_sh[0] + r_sh[0]) / 2
                mid_sh_y = (l_sh[1] + r_sh[1]) / 2
                mid_hip_x = (l_hip[0] + r_hip[0]) / 2
                mid_hip_y = (l_hip[1] + r_hip[1]) / 2
                
                # Calculate horizontal (dx) and vertical (dy) length of the Torso
                dx = abs(mid_sh_x - mid_hip_x)
                dy = abs(mid_sh_y - mid_hip_y)
                
                # Lying down: The torso is horizontal (X distance > Y distance)
                if dx > dy:
                    posture = "Lying"
                else:
                    # Torso is vertical. Are they sitting or standing?
                    torso_height = dy
                    box_height = bbox[3] - bbox[1]
                    
                    # Standing: The full body is at least 2.2 times taller than just the torso
                    if box_height > torso_height * 2.2:
                        posture = "Standing"
                    else:
                        posture = "Sitting"
                
                #fall-detect
                # Finding the center of the torso and add to history (ID'ye özel listeye ekliyoruz)
                mid_torso_y = (mid_sh_y + mid_hip_y) / 2
                people_history[person_id]['torso_y'].append(mid_torso_y)
                
                # only the last 10 frames (approx 0.3 seconds)
                if len(people_history[person_id]['torso_y']) > 10: 
                    people_history[person_id]['torso_y'].pop(0)
                    
                # vertical velocity (v_y)
                # Positive value means dropping DOWN
                if len(people_history[person_id]['torso_y']) >= 5:
                    v_y = people_history[person_id]['torso_y'][-1] - people_history[person_id]['torso_y'][0]
                else:
                    v_y = 0
                    
                # C. Trigger Fall Alarm
                # Condition 1: Dropping fast (> 40 pixels downward)
                # Condition 2: Final posture is "Lying"
                if (v_y > 40 and posture == "Lying") or (v_y > 80):
                    fall_alert_timer = 30  # Keep alert on screen for 30 frames
                    print(f"Fall detected for ID {person_id}! Speed: {v_y}")


            elif l_sh[2] > 0.3 or r_sh[2] > 0.3:
                # If hips are completely hidden (e.g., sitting behind a desk)
                posture = "Sitting (Upper Body Only)"

            # 4. WAVING DETECTION
            waving_status = "No"
            if r_wrist[2] > 0.3 and r_sh[2] > 0.3:
                if r_wrist[1] < r_sh[1]: # Hand is above shoulder
                    # Kişinin kendi history listesine ekleme yapıyoruz
                    people_history[person_id]['hand_x'].append(r_wrist[0])
                    if len(people_history[person_id]['hand_x']) > 40: 
                        people_history[person_id]['hand_x'].pop(0)
                    
                    if len(people_history[person_id]['hand_x']) > 10:
                        diff = max(people_history[person_id]['hand_x']) - min(people_history[person_id]['hand_x'])
                        
                        # Scale-invariant waving logic based on shoulder width
                        shoulder_width = abs(l_sh[0] - r_sh[0])
                        dynamic_threshold = max(30, shoulder_width * 0.3)
                        
                        if diff > dynamic_threshold: 
                            waving_status = "YES!"
                else:
                    people_history[person_id]['hand_x'] = []
            else:
                people_history[person_id]['hand_x'] = []

            # 5. FINAL UI
            color = (0, 0, 255) if waving_status == "YES!" else (0, 255, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"ID:{person_id} Pose: {posture}", (bbox[0], bbox[1]-10), 0, 0.6, color, 2)
            cv2.putText(frame, f"Waving: {waving_status}", (bbox[0], bbox[1]+25), 0, 0.6, (255, 255, 0), 2)

    cv2.imshow("SemuBot Keskin", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

camera.release()
cv2.destroyAllWindows()