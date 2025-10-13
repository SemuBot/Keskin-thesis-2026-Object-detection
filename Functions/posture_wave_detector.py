"""
Posture & Waving 
- Standing / Sitting / Lying 
- Waving if wrist is above shoulder and shakes a bit (kind of sketchy for now.)

"""

import argparse, time, math
from collections import deque, defaultdict
import cv2, numpy as np
from ultralytics import YOLO

# keypoints (COCO 17)
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW       = 7, 8
L_WRIST, R_WRIST       = 9, 10
L_HIP, R_HIP           = 11, 12
L_KNEE, R_KNEE         = 13, 14
L_ANKLE, R_ANKLE       = 15, 16

# simple skeleton for drawing
SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(11,12),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]

# tiny knobs
MIN_KP_CONF = 0.20           # a bit noisy kp (helps my laptop webcam)
HIST_LEN = 10                # wrist history (frames)
WAVE_MIN_X_SWING = 0.035     # minimal horizontal swing (0..1)
WAVE_MIN_TIME = 0.20         # sec

#will be updated