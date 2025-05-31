import math
import matplotlib.pyplot as plt
import numpy as np

L = 2.875
BASE_SPEED = 3.0
MAX_STEERING = 0.5
ANGLE_MARGIN = 0.1

def update_pose(x, y, theta, v, delta, dt, L):
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt
    theta += (v / L) * math.tan(delta) * dt
    return x, y, theta

def compute_control(x, y, theta, x_target, y_target):
    dx = x_target - x
    dy = y_target - y
    distance = math.hypot(dx, dy)

    # Oblicz kąt do celu
    angle_to_target = math.atan2(dy, dx)
    angle_error = angle_to_target - theta

    print(f"angle_error: {angle_error}")

    while angle_error > math.pi:
        angle_error -= 2 * math.pi
    while angle_error < -math.pi:
        angle_error += 2 * math.pi

    if abs(angle_error) > (math.pi / 2 + ANGLE_MARGIN):
        v = -BASE_SPEED * distance
        # Korekta kąta - bo jedziemy tyłem, "przesuń" błąd o pi
        if angle_error > 0:
            angle_error -= math.pi
        else:
            angle_error += math.pi
    # elif abs(angle_error) < (math.pi / 2 - ANGLE_MARGIN):
    else:
        v = BASE_SPEED * distance

    delta = math.atan2(2 * L * math.sin(angle_error) / distance, 1.0) 

    delta = max(min(delta, MAX_STEERING), -MAX_STEERING)

    return v, delta

def generate_parking_trajectory(start_pos, start_theta, goal_pos, goal_theta, num_points=100):
    """
    Generuje trajektorię parkowania między pozycją początkową a końcową.

    :param start_pos: Tuple (x, y) pozycji początkowej pojazdu.
    :param start_theta: Orientacja początkowa pojazdu w radianach.
    :param goal_pos: Tuple (x, y) pozycji końcowej (miejsce parkingowe).
    :param goal_theta: Orientacja końcowa pojazdu w radianach.
    :param L: Rozstaw osi pojazdu.
    :param num_points: Liczba punktów na trajektorii.
    :return: Lista punktów trajektorii [(x0, y0), (x1, y1), ..., (xn, yn)].
    """
    # Punkty kontrolne dla krzywej Béziera
    p0 = np.array(start_pos)
    p3 = np.array(goal_pos)

    d = np.linalg.norm(p3 - p0) / 3

    # Wektory kierunkowe na podstawie orientacji
    v0 = np.array([np.cos(start_theta), np.sin(start_theta)]) * d
    v3 = np.array([np.cos(goal_theta), np.sin(goal_theta)]) * d

    p1 = p0 + v0
    p2 = p3 - v3

    t_values = np.linspace(0, 1, num_points)
    trajectory = []
    for t in t_values:
        point = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
        trajectory.append((point[0], point[1]))

    return trajectory

def calculate_theta(start_point, end_point):
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    theta = np.arctan2(dy, dx)
    return theta
