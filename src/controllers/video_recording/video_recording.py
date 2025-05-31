from controller import Camera, Display, GPS, Keyboard, Lidar, Robot
from vehicle import Driver
import numpy as np
import math
import numpy as np
import cv2
from math import sqrt

DRIVER = Driver()
KEYBOARD = Keyboard()

XYZ = (None, None, None)
TIME_STEP = 100
UNKNOWN = 99999.99

# Line following PID
KP = 0.25
KI = 0.006
KD = 2

PID_NEED_RESET = False

# Size of the yellow line angle filter
FILTER_SIZE = 3

# ENABLE VARIOUS 'FEATURES'
ENABLE_COLLISION_AVOIDANCE = False
ENABLE_DISPLAY = False
HAS_GPS = False
HAS_CAMERA = False

# CAMERA
CAMERA = None
CAMERA_WIDTH = -1
CAMERA_HEIGHT = -1
CAMERA_FOV = -1.0

# SICK LASER
SICK = None
SICK_WIDTH = -1
SICK_RANGE = -1.0
SICK_FOV = -1.0

# SPEEDOMETER
DISPLAY = None
DISPLAY_WIDTH = 0
DISPLAY_HEIGHT = 0
SPEEDOMETER_IMAGE = None

# GPS
GPS = None
GPS_COORDS = np.zeros(3)
GPS_SPEED = 0.0

# MISC VARIABLES
SPEED = 0.0
STEERING_ANGLE = 0.0
MANUAL_STEERING = 0
AUTODRIVE = True

# FOR FILTER ANGLE
FIRST_CALL_FILTER_ANGLE = True
FILTER_ANGLE_LIST = np.zeros(FILTER_SIZE)

# FOR APPLY PID
OLD_VALUE_APPLY_PID = 0.0
INTEGRAL_APPLY_PID = 0.0

def print_help():
  print("You can drive this car!")
  print("Select the 3D window and then use the cursor keys to:")
  print("[LEFT]/[RIGHT] - steer")
  print("[UP]/[DOWN] - accelerate/slow down")


def set_autodrive(onoff):
    if (AUTODRIVE == onoff):
        return
    AUTODRIVE = onoff
    if AUTODRIVE:
        if HAS_CAMERA:
            print("switching to auto-drive...")
        else:
            print("impossible to switch auto-drive on without camera..")
    else:
        print("switching to manual drive...")
        print("hit [A] to return to auto-drive.")


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

    # if wheel_angle > 0.5:
    #     wheel_angle = 0.5
    # if wheel_angle < -0.5:
    #     wheel_angle = -0.5 
    wheel_angle = max(min(wheel_angle, 0.5), -0.5)
    STEERING_ANGLE = wheel_angle
    DRIVER.setSteeringAngle(STEERING_ANGLE)

def change_manual_steer_angle(inc):
    global MANUAL_STEERING
    set_autodrive(False)
    new_manual_steering = MANUAL_STEERING + inc
    if -25.0 <= new_manual_steering <= 25.0:
        MANUAL_STEERING = new_manual_steering
        set_steering_angle(MANUAL_STEERING * 0.02)
    print(f"{'turning left' if STEERING_ANGLE < 0 else 'turning right'} ({STEERING_ANGLE} rad)")

def check_keyboard():
    global KEYBOARD
    key = KEYBOARD.getKey()
    print(f"key: {key}")
    if key == KEYBOARD.UP:
        set_speed(SPEED + 5.0)
    elif key == KEYBOARD.DOWN:
        set_speed(SPEED - 5.0)
    elif key == KEYBOARD.RIGHT:
        change_manual_steer_angle(+1)
    elif key == KEYBOARD.LEFT:
        change_manual_steer_angle(-1)
    elif key == ord('A'):
        set_autodrive(True)

def color_diff(a, b):
    return sum(abs(a[i] - b[i]) for i in range(3))

# returns approximate angle of yellow road line
# or UNKNOWN if no pixel of yellow line visible
def process_camera_image(image):
    REF = [95, 187, 203] # road yellow (BGR format)
    sumx, pixel_count = 0, 0
    for x in range(CAMERA_HEIGHT * CAMERA_WIDTH):
        pixel = image[x*4:(x*4)+3]
        if color_diff(pixel, REF) < 30:
            sumx += x % CAMERA_WIDTH
            pixel_count += 1 # count yellow pixels
    return UNKNOWN if pixel_count == 0 else (sumx / pixel_count / CAMERA_WIDTH - 0.5) * CAMERA_FOV

def filter_angle(new_value):
    global FIRST_CALL_FILTER_ANGLE
    global FILTER_ANGLE_LIST

    if FIRST_CALL_FILTER_ANGLE or new_value == UNKNOWN:
        FIRST_CALL_FILTER_ANGLE = False
        FILTER_ANGLE_LIST = np.zeros(FILTER_SIZE)
    else:
        FILTER_ANGLE_LIST[:FILTER_SIZE-1] = FILTER_ANGLE_LIST[1:]

    if new_value == UNKNOWN:
        return UNKNOWN
    else:
        FILTER_ANGLE_LIST[FILTER_SIZE-1] = new_value
        return np.mean(FILTER_ANGLE_LIST)

# returns approximate angle of obstacle
# or UNKNOWN if no obstacle was detected
def process_sick_data(sick_data):
    HALF_AREA = 20 # check 20 degrees wide middle area
    obstacle_dist = 0.0
    sumx, collision_count = 0, 0

    for x in range(SICK_WIDTH // 2 - HALF_AREA, SICK_WIDTH // 2 + HALF_AREA):
        # print(f"x: {x}")
        range_data = sick_data[x]
        # print(f"range_data: {range_data}")
        if range_data < 20.0 and range_data > 0.0:
            sumx += x
            collision_count += 1
            obstacle_dist += range_data

    # if no obstacle was detected...
    if collision_count == 0:
        return UNKNOWN, obstacle_dist

    obstacle_dist /= collision_count
    obstacle_angle = ((sumx / collision_count) / SICK_WIDTH - 0.5) * SICK_FOV
    return obstacle_angle, obstacle_dist

# def update_display(current_speed):
#     global DISPLAY

#     NEEDLE_LENGTH = 50.0

#     # Display background
#     DISPLAY.imagePaste(SPEEDOMETER_IMAGE, 0, 0, False)

#     # Draw speedometer needle
#     if math.isnan(current_speed):
#         current_speed = 0.0
#     alpha = current_speed / 260.0 * 3.72 - 0.27
#     x = int(-NEEDLE_LENGTH * np.cos(alpha))
#     y = int(-NEEDLE_LENGTH * np.sin(alpha))
#     DISPLAY.drawLine(100, 95, 100 + x, 95 + y)

#     # Display GPS coordinates and speed
#     DISPLAY.drawText(f"GPS coords: {GPS_COORDS[0]:.1f} {GPS_COORDS[1]:.1f}", 10, 130)
#     DISPLAY.drawText(f"GPS speed: {GPS_SPEED:.1f}", 10, 140)

def compute_gps_speed():
    global GPS_SPEED, GPS_COORDS, GPS
    coords = GPS.getValues()
    speed_ms = GPS.getSpeed()
    GPS_SPEED = speed_ms * 3.6  # Convert to km/h
    GPS_COORDS = coords[:len(GPS_COORDS)]  

def applyPID(yellow_line_angle):
    global PID_NEED_RESET, OLD_VALUE_APPLY_PID, INTEGRAL_APPLY_PID

    if PID_NEED_RESET:
        OLD_VALUE_APPLY_PID = yellow_line_angle
        INTEGRAL_APPLY_PID = 0.0
        PID_NEED_RESET = False

    if math.copysign(1, yellow_line_angle) != math.copysign(1, OLD_VALUE_APPLY_PID):
        INTEGRAL_APPLY_PID = 0.0

    diff = yellow_line_angle - OLD_VALUE_APPLY_PID
    if -30 < INTEGRAL_APPLY_PID < 30:
        INTEGRAL_APPLY_PID += yellow_line_angle

    OLD_VALUE_APPLY_PID = yellow_line_angle
    return KP * yellow_line_angle + KI * INTEGRAL_APPLY_PID + KD * diff

def distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    global DRIVER
    global ENABLE_COLLISION_AVOIDANCE, SICK
    global ENABLE_DISPLAY, DISPLAY
    global HAS_GPS, GPS
    global HAS_CAMERA, CAMERA
    global STEERING_ANGLE, PID_NEED_RESET

    for j in range(DRIVER.getNumberOfDevices()):
        device = DRIVER.getDeviceByIndex(j)
        name = device.getName()
        if name == "Sick LMS 291":
            ENABLE_COLLISION_AVOIDANCE = True
            SICK = device
        elif name == "display":
            ENABLE_DISPLAY = True
            DISPLAY = device
        elif name == "gps":
            HAS_GPS = True
            GPS = device
        elif name == "camera":
            HAS_CAMERA = True
            CAMERA = device

    if HAS_CAMERA:
        CAMERA.enable(TIME_STEP)
        global CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FOV
        CAMERA_WIDTH = CAMERA.getWidth()
        CAMERA_HEIGHT = CAMERA.getHeight()
        CAMERA_FOV = CAMERA.getFov()

    if ENABLE_COLLISION_AVOIDANCE:
        SICK.enable(TIME_STEP)
        global SICK_WIDTH, SICK_RANGE, SICK_FOV
        SICK_WIDTH = SICK.getHorizontalResolution()
        SICK_RANGE = SICK.getMaxRange()
        SICK_FOV = SICK.getFov()

    if HAS_GPS:
        GPS.enable(TIME_STEP)

    if ENABLE_DISPLAY:
        global SPEEDOMETER_IMAGE
        SPEEDOMETER_IMAGE = DISPLAY.imageLoad("speedometer.png")

    # start engine
    if HAS_CAMERA:
        set_speed(20.0)

    DRIVER.setHazardFlashers(True)
    DRIVER.setDippedBeams(True)
    DRIVER.setAntifogLights(True)
    DRIVER.setWiperMode('SLOW')

    #print_help()

    # allow to switch to manual control
    KEYBOARD.enable(TIME_STEP)
    i = 0

    fps = 6000 / TIME_STEP
    duration = 120
    video_filename = "output_video_60fps_120s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (CAMERA_WIDTH, CAMERA_HEIGHT))
    frames_to_capture = int(fps * duration)  # Ilość klatek do przechwycenia
    frame_count = 0

    depth_video_filename = "output_depth_60fps_120s.avi"
    # isColor=False -> pojedynczy kanał
    out_depth = cv2.VideoWriter(depth_video_filename,
                                fourcc,
                                fps,
                                (CAMERA_WIDTH, CAMERA_HEIGHT), 
                                isColor=False)

    route = [
    (47.0, -73.0), # start point
    (46.0, 10.0), # start turning left
    (20.0, 38.0), # end turning left, drive forward from here
    (-28.0, 46.0), # crossing and turn right
    (-41, 58),
    (-37, 70),
    (-13, 95)
    ]

    which_point_next = 1
    driving_mode = "LINE_FOLLOW"
    CROSSING = False

    while DRIVER.step() != -1 and frame_count < frames_to_capture:

        if i % int(TIME_STEP / DRIVER.getBasicTimeStep()) == 0:
            camera_image, sick_data = None, None
            if HAS_CAMERA:
                
                camera_image = CAMERA.getImage()
                image = np.frombuffer(camera_image, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                out.write(image_rgb)
                rgb_filename = f"frames/{frame_count:05d}.png"
                cv2.imwrite(rgb_filename, image_rgb)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if ENABLE_COLLISION_AVOIDANCE:
                sick_data = SICK.getRangeImage()
                # konwersja listy floatów do 1D numpy (float32)
                depth_array = np.array(sick_data, dtype=np.float32)

                depth_array = np.nan_to_num(
                    depth_array,
                    nan=SICK_RANGE,
                    posinf=SICK_RANGE,
                    neginf=0.0
                ) 
                
                depth_mm = (depth_array * 1000.0).astype(np.uint16)
                # przeskalowanie do [0,255], zakładamy max zasięg SICK_RANGE
                # depth_clipped = np.clip(depth_array, 0.0, SICK_RANGE)
                # depth_norm = (depth_clipped / SICK_RANGE * 255).astype(np.uint8)
                # reshape do obrazu (1×SICK_WIDTH)
                # depth_image = depth_norm.reshape((1, SICK_WIDTH))
                # powiększenie do standardowego rozmiaru (opcjonalnie)
                # depth_resized = cv2.resize(depth_image, (CAMERA_WIDTH, CAMERA_HEIGHT))
                depth_filename = f"depth_frames/{frame_count:05d}.png"
                cv2.imwrite(depth_filename, depth_mm)
                out_depth.write(depth_mm)

            if AUTODRIVE and HAS_CAMERA:
                yellow_line_angle = filter_angle(process_camera_image(camera_image))
                obstacle_angle, obstacle_dist = process_sick_data(sick_data) if ENABLE_COLLISION_AVOIDANCE else (UNKNOWN, 0)

                if ENABLE_COLLISION_AVOIDANCE and obstacle_angle != UNKNOWN:
                    DRIVER.setBrakeIntensity(0.0)
                    obstacle_steering = STEERING_ANGLE
                    if 0.0 < obstacle_angle < 0.4:
                        obstacle_steering = STEERING_ANGLE + (obstacle_angle - 0.25) / obstacle_dist
                    elif obstacle_angle > -0.4:
                        obstacle_steering = STEERING_ANGLE + (obstacle_angle + 0.25) / obstacle_dist
                    steer = STEERING_ANGLE

                    if yellow_line_angle != UNKNOWN:
                        line_following_steering = applyPID(yellow_line_angle)
                        if obstacle_steering > 0 and line_following_steering > 0:
                            steer = max(obstacle_steering, line_following_steering)
                        elif obstacle_steering < 0 and line_following_steering < 0:
                            steer = min(obstacle_steering, line_following_steering)
                    else:
                        PID_NEED_RESET = True

                    set_steering_angle(steer)

                elif driving_mode == "LINE_FOLLOW":
                    # print(f"yellow_line_angle: {yellow_line_angle}")
                    if yellow_line_angle != UNKNOWN:
                        DRIVER.setBrakeIntensity(0.0)
                        set_steering_angle(applyPID(yellow_line_angle))
                    else:
                        DRIVER.setBrakeIntensity(0.4)
                        PID_NEED_RESET = True

                elif driving_mode == "LEFT_TURN":
                    DRIVER.setBrakeIntensity(0.0)
                    set_steering_angle(-0.1)
                    if(CROSSING):
                        set_steering_angle(0.4) 
                        CROSSING = False

                elif driving_mode == "RIGHT_TURN":
                    DRIVER.setBrakeIntensity(0.0)
                    set_steering_angle(0.1) 
                    if(CROSSING):
                        set_steering_angle(0.4) 
                        CROSSING = False
                
            if HAS_GPS:
                compute_gps_speed()
                current_pos = (GPS_COORDS[0], GPS_COORDS[1])
                # print(f"current_pos: {current_pos}")
                target_point = route[which_point_next]
                dist = distance(current_pos, target_point)

                if dist < 4.0:
                    which_point_next += 1
                if which_point_next >= len(route):
                    break

                # if which_point_next == 1:
                #     driving_mode = "LINE_FOLLOW"
                # elif which_point_next == 2:
                #     driving_mode = "LEFT_TURN"
                # elif which_point_next == 3:
                #     driving_mode = "LINE_FOLLOW"
                # elif which_point_next == 4:
                #     driving_mode = "RIGHT_TURN"
                #     CROSSING = True
                # elif which_point_next == 5:
                #     # set_steering_angle(0.0)
                #     driving_mode = "LINE_FOLLOW"
                #     CROSSING = False
                # elif which_point_next == 6:
                #     driving_mode = "RIGHT_TURN"
                # else:
                #     # set_steering_angle(0.0)
                #     driving_mode = "LINE_FOLLOW"

                # if driving_mode == "STRAIGHT":
                #     set_steering_angle(0.0)
                # if driving_mode == "LEFT_TURN":
                #     set_steering_angle(-0.1)
                # elif driving_mode == "RIGHT_TURN":
                #     set_steering_angle(0.1)

            # if ENABLE_DISPLAY:
            #     update_display(DRIVER.getCurrentSpeed())
        i += 1
        print(f"frame_count: {frame_count}, to_capture: {frames_to_capture}")

    del DRIVER
    out.release()
    out_depth.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    