## Capstone_Project
## Markerless Hand Motion Tracking for Quantification of Surgical Skills

## Description
This project utilizes Google's MediaPipe to track surgical hand motion, extracting key points of the hand's movement in real-time from recorded videos. Moreover, the team is also tracking hand motion in a markerbased system called PhaseSpace and the markerless system will be validated against the markerbased. 
The code was provided to the capstone team by the medical students of Barrow Neurological Institute. Its an ongoing project and the capstone team is currently working on refining another algorithm for markerless hand motion tracking in 3D. 

## Features
- Tracks 21 key points on the hand, including fingertips and joints.
- Outputs 2D coordinates of hand landmarks in pixels.
- Supports live camera input or pre-recorded video files.
- Generates CSV files containing tracking data for further analysis.

## Requirements
- Python
- Required Python packages:
  - MediaPipe
  - Opencv-python
  - Numpy

## The code for markerless 2D tracking can be found in the master branch of this repository.
