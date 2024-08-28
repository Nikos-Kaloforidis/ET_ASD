from . import Visuals
import cv2 as cv
import numpy as np

screen_width, screen_height = Visuals.getScreenDim()

print(screen_width, ",", screen_height)


Calibration_Points = [
    (int(screen_width / 2), int(screen_height / 2)),  # Center point
    (100, 100),  # Top left point
    ((screen_width - 100), 100),  # Top right Point
    (100, (screen_height - 100)),  # Bottom left Point
    ((screen_width - 100), (screen_height - 100)),  # Bottom Right Points
]


def show_calibration_points(image, calib_point, gaze_coords=None, blur_radius=20):
    cv.namedWindow("Calibrating", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Calibrating", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.circle(image, calib_point, 30, (0, 255, 0), -1)

    if gaze_coords:

        for x, y in gaze_coords:
            mask = np.zeros_like(image)
            cv.circle(mask, (int(x), int(y)), blur_radius, (255, 0, 0), -1)
            blurred_mask = cv.GaussianBlur(
                mask, (2 * blur_radius + 1, 2 * blur_radius + 1), blur_radius
            )
            image = cv.addWeighted(image, 1.0, blurred_mask, 0.5, 0)

    return image


# Calibrates the eye tracker by comparing the gaze to fix points and then calculating
# the  homogenous matrix
def calibrate_eye_tracker(cam, faceMesh):
    eyes_coordinates = []

    cv.namedWindow("Calibrating", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Calibrating", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    for point in Calibration_Points:
        data_samples = []

        for _ in range(30):
            ret, frame = cam.read()

            if not ret:
                print("Didn’t capture a frame!")
                continue

            flip_frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(flip_frame, cv.COLOR_BGR2RGB)
            output = faceMesh.process(rgb_frame)
            landmarks_points = output.multi_face_landmarks

            if landmarks_points:
                landmarks = landmarks_points[0].landmark

                eye_1 = [
                    (landmark_eye.x, landmark_eye.y)
                    for landmark_eye in landmarks[473:479]
                ]
                eye_2 = [
                    (landmark_eye.x, landmark_eye.y)
                    for landmark_eye in landmarks[468:473]
                ]

                avg_x = (sum(x for x, _ in eye_1) + sum(x for x, _ in eye_2)) / (
                    len(eye_1) + len(eye_2)
                )
                avg_y = (sum(y for _, y in eye_1) + sum(y for _, y in eye_2)) / (
                    len(eye_1) + len(eye_2)
                )

                data_samples.append((avg_x, avg_y))

                screen_x = int(avg_x * screen_width)
                screen_y = int(avg_y * screen_height)

                calib_image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
                calib_image = show_calibration_points(
                    calib_image, point, [(screen_x, screen_y)]
                )

                cv.imshow("Calibrating", calib_image)
                cv.waitKey(1)

        if data_samples:
            avg_coords = np.mean(data_samples, axis=0)
            eyes_coordinates.append(avg_coords)
        else:
            print(f"No valid eye coordinates for point {point}.")

    if len(eyes_coordinates) < 4:
        print("Not enough calibration data. Calibration failed.")
        cv.destroyWindow("Calibrating")
        return None

    source_points = np.array(eyes_coordinates, dtype=np.float32)
    destination_points = np.array(Calibration_Points, dtype=np.float32)

    transform_matrix, _ = cv.findHomography(source_points, destination_points)

    print("Calibration finished!")

    cv.destroyWindow("Calibrating")

    return transform_matrix
