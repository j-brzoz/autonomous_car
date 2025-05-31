from controller import Camera, GPS, Lidar
from vehicle import Driver
import numpy as np
import math
import cv2
from math import sqrt
from bicycle_model import compute_control, calculate_theta, generate_parking_trajectory
import matplotlib.pyplot as plt

DRIVER = Driver()

XYZ = (None, None, None)
TIME_STEP = 100
UNKNOWN = 99999.99

PID_NEED_RESET = False

# ENABLE VARIOUS 'FEATURES'
ENABLE_COLLISION_AVOIDANCE = False
CAMERA = None
CAMERA_WIDTH = -1
CAMERA_HEIGHT = -1
CAMERA_FOV = -1.0

# CAMERA_REAR
CAMERA_REAR = None
CAMERA_REAR_WIDTH = -1
CAMERA_REAR_HEIGHT = -1
CAMERA_REAR_FOV = -1.0

# SICK LASER
SICK = None
SICK_WIDTH = -1
SICK_RANGE = -1.0
SICK_FOV = -1.0


# GPS
GPS = None
GPS_COORDS = np.zeros(3)
GPS_SPEED = 0.0

# GYROSCOPE
GYRO = None
GYRO_HEADING = 1.5
GYRO_LAST_UPDATE_TIME = 0

#compass
COMPASS = None
HAS_COMPASS = False

# MISC VARIABLES
SPEED = 0.0
STEERING_ANGLE = 0.0
WHEELBASE = 2.875

# set target speed
def set_speed(kmh):
    global SPEED
    # max speed
    if (kmh > 250.0):
        kmh = 250.0

    SPEED = kmh
    print(f"setting speed to {kmh} km/h", )
    DRIVER.setCruisingSpeed(kmh)

# positive: turn right, negative: turn left
def set_steering_angle(wheel_angle):
    global STEERING_ANGLE, DRIVER
    if wheel_angle - STEERING_ANGLE > 0.1:
        wheel_angle = STEERING_ANGLE + 0.1
    if wheel_angle - STEERING_ANGLE < -0.1:
        wheel_angle = STEERING_ANGLE - 0.1

    wheel_angle = max(min(wheel_angle, 0.5), -0.5)
    STEERING_ANGLE = wheel_angle
    DRIVER.setSteeringAngle(STEERING_ANGLE)

def compute_gps_speed():
    global GPS_SPEED, GPS_COORDS, GPS
    coords = GPS.getValues()
    speed_ms = GPS.getSpeed()
    GPS_SPEED = speed_ms * 3.6  # Convert to km/h
    GPS_COORDS = coords[:len(GPS_COORDS)]  

def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

BASE_SPEED = 5.0
MAX_STEERING = 0.5

def main():
    global DRIVER
    global ENABLE_COLLISION_AVOIDANCE, SICK
    global HAS_GPS, GPS
    global HAS_CAMERA, CAMERA, CAMERA_REAR
    global STEERING_ANGLE, PID_NEED_RESET
    global GYRO

    for j in range(DRIVER.getNumberOfDevices()):
        device = DRIVER.getDeviceByIndex(j)
        name = device.getName()
        if name == "Sick LMS 291":
            ENABLE_COLLISION_AVOIDANCE = True
            SICK = device
        elif name == "gps":
            HAS_GPS = True
            GPS = device
        elif name == "camera":
            HAS_CAMERA = True
            CAMERA = device
        elif name == "camera_rear":
            HAS_CAMERA = True
            CAMERA_REAR = device
        elif name == "gyro":
            GYRO = device
        elif name == "compass":
            print("compass")
            HAS_COMPASS = True
            COMPASS = device

    if HAS_CAMERA:
        CAMERA.enable(TIME_STEP)
        global CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV
        CAMERA_WIDTH = CAMERA.getWidth()
        CAMERA_HEIGHT = CAMERA.getHeight()
        CAMERA_FOV = CAMERA.getFov()

        CAMERA_REAR.enable(TIME_STEP)
        global CAMERA_REAR_WIDTH, CAMERA_REAR_HEIGHT, CAMERA_REAR_FOV
        CAMERA_REAR_WIDTH = CAMERA_REAR.getWidth()
        CAMERA_REAR_HEIGHT = CAMERA_REAR.getHeight()
        CAMERA_REAR_FOV = CAMERA_REAR.getFov()

    if ENABLE_COLLISION_AVOIDANCE:
        SICK.enable(TIME_STEP)
        global SICK_WIDTH, SICK_RANGE, SICK_FOV
        SICK_WIDTH = SICK.getHorizontalResolution()
        SICK_RANGE = SICK.getMaxRange()
        SICK_FOV = SICK.getFov()

    if HAS_GPS:
        GPS.enable(TIME_STEP)
        GYRO.enable(TIME_STEP)
        COMPASS.enable(TIME_STEP)

    # start engine
    if HAS_CAMERA:
        set_speed(BASE_SPEED)

    DRIVER.setHazardFlashers(True)
    DRIVER.setDippedBeams(True)
    DRIVER.setAntifogLights(True)
    DRIVER.setWiperMode('SLOW')

    i = 0

    start_point = (314.0, -472.0)
    end_point = (317.0, -477.0)

    start_theta = calculate_theta(start_point, end_point)
    goal_theta = start_theta
    print(f"start_theta: {start_theta}")

    trajectory = generate_parking_trajectory(start_point, start_theta, end_point, goal_theta)
    x_target = end_point[0]
    y_target = end_point[1]

    theta = 1.5
    current_target_index = 1
    alpha = 0.95

    while DRIVER.step() != -1:

        if i % int(TIME_STEP / DRIVER.getBasicTimeStep()) == 0:
            if HAS_GPS and i > 1:
                compute_gps_speed()
                current_pos = (GPS_COORDS[0], GPS_COORDS[1])
                x = GPS_COORDS[0]
                y = GPS_COORDS[1]
                
                compass_values = COMPASS.getValues()
                compass_theta = math.atan2(compass_values[0], compass_values[2])

                gyro_values = GYRO.getValues()
                theta_gyro = theta + gyro_values[1] * (TIME_STEP / 1000.0)
                theta = alpha * theta_gyro + (1 - alpha) * compass_theta

                while theta > math.pi:
                    theta -= 2 * math.pi
                while theta < -math.pi:
                    theta += 2 * math.pi


                x_target, y_target = trajectory[current_target_index]
                v, delta = compute_control(x, y, theta, x_target, y_target)

                set_speed(v)
                set_steering_angle(delta)

                if math.hypot(x - x_target, y - y_target) < 0.2:
                    current_target_index += 1
                    if current_target_index >= len(trajectory):
                        print("Zaparkowano")
                        set_speed(0.0)
                        set_steering_angle(0.0)
                        break

        i += 1

    del DRIVER

if __name__ == "__main__":
    main()
    