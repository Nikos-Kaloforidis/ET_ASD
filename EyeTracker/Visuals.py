import pyautogui
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def start_window():
    screen_width, screen_height = pyautogui.size()

    image = np.zeros((screen_width, screen_height, 3), dtype=np.uint8)

    text_size = cv.getTextSize(
        "Press spacebar to continue", cv.FONT_HERSHEY_COMPLEX, 1, 1
    )[0]

    text_x_pos = (image.shape[1] - text_size[0]) // 2
    text_y_pos = image.shape[0] - 100

    cv.namedWindow("Message", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Message", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    blink = True
    cv.putText(
        image,
        "In order to calibrate look the green dots",
        (200, 150),
        cv.FONT_HERSHEY_COMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    while True:
        if blink:
            cv.putText(
                image,
                "Press spacebar to continue",
                (text_x_pos, text_y_pos),
                cv.FONT_HERSHEY_COMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        else:
            cv.rectangle(
                image,
                (text_x_pos, text_y_pos - text_size[1]),
                (text_x_pos + text_size[0], text_y_pos + 2),
                (0, 0, 0),
                -1,
            )

        cv.imshow("Message", image)

        blink = not blink
        if cv.waitKey(500) & 0xFF == ord(" "):
            break
    cv.destroyWindow("Message")


def getScreenDim():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


def add_indicators(image, gaze_points, blur_radius=80):
    for x, y in gaze_points:
        mask = np.zeros_like(image)
        cv.circle(mask, (int(x), int(y)), blur_radius, (0, 255, 0), -1)
        blurred_mask = cv.GaussianBlur(
            mask, (2 * blur_radius + 1, 2 * blur_radius + 1), blur_radius
        )
        image = cv.addWeighted(image, 1.0, blurred_mask, 0.5, 0)
    return image


# This function will use speed acceleration and jerk  as rgb values for the plot
def dynamics_to_rgb(velocity, acceleration, jerk):
    r = velocity
    g = acceleration
    b = jerk
    return (r, g, b)


# This function will print  and save the plot of eye movement on the screen
# with a black background
def createImage(x_movement, y_movement, speed_cap, accel_cap, jerk_cap):

    screen_width, screen_height = getScreenDim()

    x_movement = np.clip(
        x_movement, 0, screen_width
    )  # Clip to prevent out-of-bounds values
    y_movement = np.clip(y_movement, 0, screen_height)

    # Normalize coordinates
    x_movement_normalized = (x_movement / screen_width) * 640
    y_movement_normalized = (y_movement / screen_height) * 480

    plt.figure(figsize=(640 / 100, 480 / 100), dpi=100, facecolor="black")

    plt.xlim(0, 640)
    plt.ylim(480, 0)

    plt.axis("off")

    for i in range(1, len(x_movement)):
        x0, y0 = x_movement_normalized[i - 1], y_movement_normalized[i - 1]
        x1, y1 = x_movement_normalized[i], y_movement_normalized[i]

        color = dynamics_to_rgb(speed_cap[i - 1], accel_cap[i - 1], jerk_cap[i - 1])

        plt.plot([x0, x1], [y0, y1], color=color, linestyle="-")

    plt.gca().invert_yaxis()

    plt.savefig("test_image.png")
    plt.show()
