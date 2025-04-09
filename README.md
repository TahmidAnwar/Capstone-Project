## Capstone_Project
## Markerless Hand Motion Tracking for Quantification of Surgical Skills

## Description
This project utilizes Google's MediaPipe and Intel RealSense Depth Camera to track surgical hand motion in 3D, extracting key points of the hand's movement in real-time from recorded videos. Moreover, the hand motion is also tracked in a markerbased system called PhaseSpace and the markerless system will be validated against the markerbased. 
The 2D code was provided to the capstone team by the medical students of Barrow Neurological Institute. The capstone team modified the algorithm for markerless hand motion tracking in 3D. 

## Features
- Uses OpenCV and MediaPipe to track 3D hand movements through the Intel RealSense D435 Depth Camera.
- Outputs csv file of time interval of each frame and theb 3D trajectory points based on the 21 hand landmark model of the MediaPipe 
- Supports pre-recorded video files.
- 
## Requirements
- Python
- Required Python packages:
  - MediaPipe
  - OpenCV-python
  - Numpy
  - Pyrealsense2

**The codes for markerless 2D and 3D tracking can be found in the master branch of this repository**
