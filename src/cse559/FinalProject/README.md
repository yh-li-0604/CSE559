# Real‑Time Face Detection and Privacy‑Preserving Blurring in Video Streams

### Team members: Zeyuan Zong, Yunhe Li

## Abstract
We are developing a program that automatically detects and blurs all human faces in both live webcam streams and prerecorded video footage. The system takes video input and outputs the same format, but with each detected face annotated by a bounding box and blurred using a Gaussian filter. To evaluate the system, we aim to measure three key aspects: (1) how accurately each bounding box localizes a face, (2) whether all visible faces are detected, and (3) how much of each face is successfully covered by the blur. Specifically, we will compute metrics such as mAP@0.5 IoU to assess localization quality, Recall and F1 score to evaluate detection completeness, and a face coverage ratio to determine the proportion of the face area that is correctly blurred. We will also assess the system’s robustness by testing its performance across different facial poses and viewing conditions to ensure consistent detection of human facial features.

rye run python -m pip install dlib --no-build-isolation