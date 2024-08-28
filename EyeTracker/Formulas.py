import numpy as np


# This function calculates the points that will be used to create the image
def calcValidPoints(x_movement, y_movement, num_points):

    # If collected data < number of points return collected data
    if num_points >= len(x_movement):
        print("Not enough points")
        return x_movement, y_movement

    x_valid = []
    y_valid = []

    # Calculate the size of the step
    step = len(x_movement) / num_points

    for i in range(num_points):
        # Calculate the int step of the index
        index = int(i * step)

        x_valid.append(x_movement[index])
        y_valid.append(y_movement[index])

    return x_valid, y_valid


# Calculates  the cap as  a quarter of the diagonal of the screen
def calculateCap(screen_width, screen_height):
    diagonal_length = np.sqrt(screen_width**2 + screen_height**2)
    cap = diagonal_length / 4
    return cap


def normalizeToCap(value, cap):
    return min(value / cap, 1.0)


# Calculates speed  with the euclidian distance of 2 consecutive points and then
# divide it by time_frame which is the time between 2 frames
def calculate_speed(x_movement, y_movement, time_frame):
    speed_of_gaze = []
    for i in range(1, len(x_movement)):

        cur_x = x_movement[i]
        cur_y = y_movement[i]

        prev_x = x_movement[i - 1]
        prev_y = y_movement[i - 1]

        # calculate euclidian distance
        dist = np.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
        cur_speed = dist / time_frame
        speed_of_gaze.append(cur_speed)

    return speed_of_gaze


# Calculates the acceleration as the  derivative of speed
def calculate_acceleration(speed_of_gaze, time_frame):
    acceleration_of_gaze = []
    for i in range(1, len(speed_of_gaze)):

        cur_acceleration = abs(speed_of_gaze[i] - speed_of_gaze[i - 1]) / time_frame
        acceleration_of_gaze.append(cur_acceleration)

    return acceleration_of_gaze


# Calculates the jerk  as the  derivative of acceleration
def calculate_jerk(acceleration_of_gaze, time_frame):
    jerk_of_gaze = []

    for i in range(1, len(acceleration_of_gaze)):

        cur_jerk = (
            abs(acceleration_of_gaze[i] - acceleration_of_gaze[i - 1]) / time_frame
        )
        jerk_of_gaze.append(cur_jerk)

    return jerk_of_gaze
