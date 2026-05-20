#  Keskin-thesis-2026-Object-detection

A visual recognition and object detection module for **SemuBot**, developed at the University of Tartu.

This repository contains YOLO-based scripts for detecting **people, objects, postures, and waving gestures**.
The goal is to give SemuBot simple visual awareness.

## Features
* **Real-Time Human Pose Estimation:** Utilizes YOLOv8 to extract 17 COCO keypoints.
* **Posture Classification:** Custom heuristic logic to classify subjects as *Standing*, *Sitting*, or *Laying* (opportunistic safety monitoring).
* **Dynamic Gesture Detection:** Identifies waving gestures for help requests using scale-invariant dynamic shoulder-width thresholding.
* **Edge-Optimized:** Designed to run efficiently on local edge hardware (~30 FPS).

## Hardware and Software Requirements
* Intel RealSense D435i Depth Camera (or standard webcam for basic testing)
* Standard Edge/Local CPU (e.g., AMD Ryzen 7 or equivalent)

All dependencies are listed in [`requirements.txt`].  
To install:

bash --> pip install -r requirements.txt

