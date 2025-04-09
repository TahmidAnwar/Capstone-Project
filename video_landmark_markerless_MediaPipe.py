import cv2
import mediapipe as mp
import csv

user = input("Enter username: ")

def main():
    # Initialize MediaPipe Hands model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=1,
                           min_detection_confidence=0.2, min_tracking_confidence=0.2)
    mp_drawing = mp.solutions.drawing_utils

    # Define output filename
    directory = r"C:\Users\anwar\Handtracking\Output"
    output_video = directory + user + "_output" + ".mp4"
    landmark_csv = directory + user + "_twohand_landmarks_run4" + ".csv"

    # Read the video file
    video_file = r"C:\Users\anwar\OneDrive\Documents\ASU\fall24\bme417\MediaPipe\python code\video_with_marker_on_camera_run4.mp4"
    cap = cv2.VideoCapture(video_file)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Output video frame rate

    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_count = -1

    # Open CSV file for writing
    with open(landmark_csv, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # csv_writer.writerow(['Frame', 'Hand Side', 'Landmark Number', 'X Coordinate', 'Y Coordinate'])

    # Loop through each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Increment frame count
            frame_count += 1

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            results = hands.process(rgb_frame)

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                # Loop through each detected hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # Determine hand side (left or right) based on landmark positions
                    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                        hand_side = "Left"
                    else:
                        hand_side = "Right"

                    # Loop through each landmark
                    for lm_id, landmark in enumerate(hand_landmarks.landmark):
                        # Get the coordinates of the landmark
                        height, width, _ = frame.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1))

                        # Write data to CSV file
                        csv_writer.writerow([frame_count / 30, lm_id, cx, cy, hand_side])

                        # Add timestamp to the frame
                        total_seconds = frame_count / 30
                        hh = int(total_seconds// 3600)
                        mm = int((total_seconds % 3600) // 60)
                        ss = int(total_seconds % 60)
                        subsec = frame_count % 30
                        timestamp = "{:02d}:{:02d}:{:02d}:{:02d}".format(hh, mm, ss, subsec)
                        cv2.putText(frame, timestamp, (100, frame.shape[0] - 125),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                        # Print the coordinates of the landmark
                        # print(f"{hand_side} Landmark {lm_id}: ({cx}, {cy})")

            # Write frame to output video
            out.write(frame)

            # Display the frame
            cv2.imshow('Frame', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

        # Release video writer
        out.release()


if __name__ == "__main__":
    main()
