import EyeTracker.EyeTrackerDataCollect as dc
from EyeTracker import Formulas, Visuals
import cv2 as cv
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
from joblib import load
from PIL import Image


print("Opening camera...")

cam = cv.VideoCapture(0)

if not cam.isOpened():
    print("Error opening camera!")
    exit()
else:
    print("Camera opened successfully!")


faceMesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

print("Calibrating Eye Tracker...")
Visuals.start_window()
tranform_matrix = dc.calibrate_eye_tracker(cam, faceMesh)
print("Generated Transform Matrix: ", tranform_matrix)

cv.waitKey(2000)

# Opens video for testing the subject
print("Opening Video...")
video = cv.VideoCapture("test_video.mp4")

if not video.isOpened():
    print("Error opening video!")
    exit()
else:
    print("Video opened successfully!")

gaze_coordinates, windowHeight, windowWidth, video_length = dc.GatherData(
    cam, faceMesh, video, tranform_matrix
)

x_movement = []
y_movement = []

cap = Formulas.calculateCap(windowWidth, windowHeight)

for x, y in gaze_coordinates:
    x_movement.append(x)
    y_movement.append(y)

x_valid, y_valid = Formulas.calcValidPoints(x_movement, y_movement, 200)

frames = video_length / len(x_valid)
time_frame = 1 / frames

speed_of_gaze = Formulas.calculate_speed(x_valid, y_valid, time_frame)

acceleration_of_gaze = Formulas.calculate_acceleration(speed_of_gaze, time_frame)

jerk_of_gaze = Formulas.calculate_jerk(acceleration_of_gaze, time_frame)

speed_capped = []
acceleration_capped = []
jerk_capped = []

for s in speed_of_gaze:
    speed_capped.append(Formulas.normalizeToCap(s, cap))

for a in acceleration_of_gaze:
    acceleration_capped.append(Formulas.normalizeToCap(a, cap))

for j in jerk_of_gaze:
    jerk_capped.append(Formulas.normalizeToCap(j, cap))

acceleration_capped = [0] + acceleration_capped
jerk_capped = [0, 0] + jerk_capped


print("\nCreating image from data...")
dc.createImage(x_valid, y_valid, speed_capped, acceleration_capped, jerk_capped)


def preproccess(image):
    image = np.array(image)
    normalize_image = image / 255.0
    if normalize_image.dtype == np.float64:
        img_32f = normalize_image.astype(np.float32)
        grayscale_image = cv.cvtColor(img_32f, cv.COLOR_BGR2GRAY)

    elif normalize_image.dtype == np.uint16:
        img_32f = normalize_image.astype(np.float32)
        grayscale_image = cv.cvtColor(img_32f, cv.COLOR_BGR2GRAY)

    elif normalize_image.dtype == np.float32 or normalize_image.dtype == np.uint8:
        grayscale_image = cv.cvtColor(normalize_image, cv.COLOR_BGR2GRAY)

    else:
        print(f"Unsupported image depth: {normalize_image.dtype}")

    resize = cv.resize(grayscale_image, (225, 225))
    flatten = resize.flatten()

    return flatten


img = Image.open("test_image.png")
print("Image loaded!")

image = preproccess(img)
print("Image preproccesed!")

image = image.reshape(1, -1)

model = load("random_forest_model.joblib")
print("Model loaded!")

predict = model.predict(image)
proba = model.predict_proba(image)
proba = np.array(proba)

print("=============================================================")
print("\t\tPrediction: ", predict, "\n\t\tProbability: ", np.max(proba))
print("=============================================================")
