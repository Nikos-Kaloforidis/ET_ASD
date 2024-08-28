import cv2 as cv
import numpy as np

from . import Visuals

gaze_coordinates = []


# Transforms the gaze to screen coordinates
def gazeToScreen(x_coor, y_coor, tranform_matrix):

    gaze_point = np.array([[x_coor, y_coor]], dtype=np.float32).reshape(-1, 1, 2)
    screenCoor = cv.perspectiveTransform(gaze_point, tranform_matrix)

    return int(screenCoor[0][0][0]), int(screenCoor[0][0][1])


# This function  will use opencv to capture video from camera then
# create a face mesh with mediapipe and detect 5 key points for each eye
# At the end it will calculate  gaze as the average of  the 2 eyes coordinates in each frame
def GatherData(cam, faceMesh, video, transform_matrix):

    window_height = 0
    window_width = 0
    frame_count = 0
    screen_width, screen_height = Visuals.getScreenDim()

    cv.namedWindow("Video", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Video", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    while True:
        _, frame = cam.read()
        ret, video_frame = video.read()

        # Exiting the program when video ends
        if not ret:
            print("End of video or error encountered.")
            break

        frame_count += 1

        # opens the window for video player
        cv.imshow("Video", video_frame)
        # cv.moveWindow("Video", 10, 10)
        # cv.setWindowProperty('Video', cv.WND_PROP_TOPMOST, 1)

        eye_1_coordinates = []
        eye_2_coordinates = []

        frame = cv.flip(frame, 1)

        # Kanei rgb to frame
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        output = faceMesh.process(rgb_frame)

        # Briskei ta landmarks tou prosopou
        landmarks_points = output.multi_face_landmarks

        # Diastaseis parathirou
        window_height, window_width, _ = frame.shape

        if landmarks_points:
            # Landmarks tou protou prosopou
            landmarks = landmarks_points[0].landmark

            # Epilegei ta landmark ton mation
            for landmark_eye_1 in landmarks[473:479]:
                eye_1_x = int(landmark_eye_1.x * window_width)
                eye_1_y = int(landmark_eye_1.y * window_height)
                eye_1_coordinates.append((eye_1_x, eye_1_y))
                cv.circle(frame, (eye_1_x, eye_1_y), 3, (255, 0, 0))

            for landmark_eye_2 in landmarks[468:473]:
                eye_2_x = int(landmark_eye_2.x * window_width)
                eye_2_y = int(landmark_eye_2.y * window_height)
                eye_2_coordinates.append((eye_2_x, eye_2_y))

                cv.circle(frame, (eye_2_x, eye_2_y), 3, (0, 255, 0))
            # Calculates the gaze relative to frame
            if eye_1_coordinates and eye_2_coordinates:
                average_x = (
                    sum(x for x, _ in eye_1_coordinates)
                    + sum(x for x, _ in eye_2_coordinates)
                ) / (len(eye_1_coordinates) + len(eye_2_coordinates))
                average_y = (
                    sum(y for _, y in eye_1_coordinates)
                    + sum(y for _, y in eye_2_coordinates)
                ) / (len(eye_1_coordinates) + len(eye_2_coordinates))

                # Makes the gaze relative to screen
                screen_x, screen_y = gazeToScreen(
                    average_x / window_width,
                    average_y / window_height,
                    transform_matrix,
                )
                # Ensures that  gaze coordinates do not exceed screen limits
                if screen_x < 0:
                    screen_x = 0
                elif screen_x >= screen_width:
                    screen_x = screen_width - 1

                if screen_y < 0:
                    screen_y = 0
                elif screen_y >= screen_height:
                    screen_y = screen_height - 1

                gaze_coordinates.append((screen_x, screen_y))
                print("Gaze: ", screen_x, screen_y)

        if len(gaze_coordinates) > 0:
            video_frame = Visuals.add_indicators(video_frame, [gaze_coordinates[-1]])

        # Display gaze coordinates on the frame
        if len(gaze_coordinates) > 0:
            latest_x, latest_y = gaze_coordinates[-1]
            cv.putText(
                frame,
                f"Gaze: ({latest_x}, {latest_y})",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv.imshow("EyeTracker", frame)

        # Waits for q to be pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Close Camera and all windows
    cam.release()
    cv.destroyAllWindows()

    return gaze_coordinates, window_height, window_width, frame_count
