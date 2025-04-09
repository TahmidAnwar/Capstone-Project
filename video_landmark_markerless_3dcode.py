import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import csv
import os
import time

# Font settings for visualization
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 50, 255)
thickness = 1

# ====== Check if .bag file exists ======
bag_file = r"C:\Users\anwar\OneDrive\Documents\ashton_4thapril_calib+suture7.bag"
if not os.path.exists(bag_file):
    print(f"Error: .bag file not found at {bag_file}")
    exit(1)

print(f"Using .bag file: {bag_file}")

# ====== Realsense Configuration ======
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)  # Don't loop

try:
    profile = pipeline.start(config)

    # Get frame dimensions for VideoWriter
    first_frameset = pipeline.wait_for_frames()
    first_color_frame = first_frameset.get_color_frame()
    frame_width = int(first_color_frame.width)
    frame_height = int(first_color_frame.height)

    # Initialize VideoWriter
    output_video = cv2.VideoWriter("hand_tracking_output.avi",
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   30, (frame_width, frame_height))

    time.sleep(1)  # Small delay to allow initialization
except Exception as e:
    print(f"Error: Could not start pipeline - {e}")
    exit(1)

align_to = rs.stream.color  # alignment
align = rs.align(align_to)

# ====== Get Depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
try:
    depth_scale = depth_sensor.get_depth_scale()
except Exception:
    print("Warning: Could not get depth scale, using default 0.001")
    depth_scale = 0.001  # Default value

print(f"Depth Scale: {depth_scale}")

clipping_distance_in_meters = 2  # Objects beyond 2 meters will be ignored
clipping_distance = clipping_distance_in_meters / depth_scale

# ====== Mediapipe Initialization ======
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, model_complexity=1,
                      min_detection_confidence=0.2, min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

# ====== Initialize CSV File ======
csv_filename = "output_coordinates.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time Interval (s)", "Hand", "Landmark Index", "X (mm)", "Y (mm)", "Z (mm)"])

previous_time = None

while True:
    current_time = time.perf_counter()

    try:
        frames = pipeline.wait_for_frames()
    except RuntimeError:
        print("End of .bag file reached.")
        break

    if not frames:
        print("No more frames available.")
        break

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    color_image = cv2.flip(color_image, 1)
    depth_image = cv2.flip(depth_image, 1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = hands.process(color_images_rgb)

    time_interval = 0.0  # Default value for the first frame
    if previous_time is not None:
        time_interval = current_time - previous_time
    if results.multi_hand_landmarks:
        for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
            original_label = results.multi_handedness[hand_idx].classification[0].label
            hand_label = "Right" if original_label == "Left" else "Left"
            mpDraw.draw_landmarks(color_image, handLms, mpHands.HAND_CONNECTIONS)

            for idx, landmark in enumerate(handLms.landmark):
                x_px = int(landmark.x * depth_image.shape[1])
                y_px = int(landmark.y * depth_image.shape[0])
                x_px = max(0, min(x_px, depth_image.shape[1] - 1))
                y_px = max(0, min(y_px, depth_image.shape[0] - 1))

                depth_value = depth_image[y_px, x_px] * depth_scale

                depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().get_intrinsics()
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_px, y_px], depth_value)
                
                X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000

                print(f"{hand_label} Hand - Landmark {idx}: (X: {X_mm:.3f} mm, Y: {Y_mm:.3f} mm, Z: {Z_mm:.3f} mm)")

                coord_text = f"L{idx}: ({x_px}, {y_px}, {depth_value * 1000:.3f}m)"
                color_image = cv2.putText(color_image, coord_text, (x_px, y_px), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([time_interval, hand_label, idx, X_mm, Y_mm, Z_mm])

    time_text = f"Time: {time_interval:.2f}s"
    cv2.putText(color_image, time_text, (frame_width - 150, frame_height - 20),
                font, fontScale, (255, 255, 255), thickness, cv2.LINE_AA)

    output_video.write(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Hand Tracking", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break
    previous_time = current_time

pipeline.stop()
output_video.release()
cv2.destroyAllWindows()
print(f"Processing Complete. Output saved in {csv_filename}")
