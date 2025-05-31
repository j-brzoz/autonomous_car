import numpy as np
import cv2


class VehicleController:
    def __init__(self, pure_pursuit, pid, driver, max_speed=50.0, min_speed=0.0, max_steering=0.5):
        self.pure_pursuit = pure_pursuit
        self.pid = pid
        self.driver = driver
        self.target_speed = max_speed
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_steering = max_steering
        self.can_go = True
        self.last_obstacle_right = False
        self.is_passing = False
        self.obstacle_front_count = 0

    def update(self, distances, center_mask, camera_width, camera_height,
               num_segments, horizontal_fov, obstacles, image, visualize=False):
        # Determine target angles (alphas) for both lanes
        alphas = self._get_target_alpha(
            distances, center_mask, camera_width, camera_height,
            num_segments, horizontal_fov, image, visualize
        )

        if self.last_obstacle_right and not obstacles["right"]:
            self.is_passing = False
            self.obstacle_front_count = 0

        if obstacles["front"]:
            self.obstacle_front_count += 1

        if self.obstacle_front_count >= 3:
            self.is_passing = True

        left_alpha, right_alpha = alphas[1], alphas[2]

        if left_alpha is not None and right_alpha is not None:
            if self.is_passing:
                if not obstacles["left"]:
                    self.pure_pursuit.alpha = left_alpha
                    self.can_go = True
                else:
                    self.can_go = False
            else:
                self.can_go = True
                self.pure_pursuit.alpha = right_alpha

        elif left_alpha is None and right_alpha is not None:
            self.can_go = not obstacles["front"]
            if self.can_go:
                self.pure_pursuit.alpha = right_alpha

        elif right_alpha is None and left_alpha is not None:
            self.can_go = not obstacles["front"]
            if self.can_go:
                self.pure_pursuit.alpha = left_alpha

        else:
            self.can_go = not obstacles["front"]
            if self.can_go:
                self.pure_pursuit.alpha = left_alpha

        print("last_obstacle_right:", self.last_obstacle_right, "| is_passing:", self.is_passing)

        # Update target speed
        self.target_speed = self._set_target_speed(self.driver.getSteeringAngle())

        # Compute control actions
        speed = self.pid.calculate_speed(self.driver.getCurrentSpeed(), self.target_speed)
        steering_angle = self.pure_pursuit.calculate_steering_angle(self.driver.getCurrentSpeed())

        # Apply control
        self.driver.setCruisingSpeed(speed)
        self.driver.setSteeringAngle(steering_angle)

        self.last_obstacle_right = obstacles["right"]

    def _set_target_speed(self, current_angle):
        if not self.can_go:
            return 0.0
        if np.isnan(current_angle):
            return self.min_speed

        factor = (abs(current_angle) / self.max_steering) ** (1 / 1.75)
        factor = np.clip(factor, 0.0, 0.9)
        return self.max_speed * (1 - factor)

    def _get_target_alpha(self, distances, center_mask, camera_width, camera_height,
                          num_segments, horizontal_fov, image, visualize=False):
        seg_w = camera_width / num_segments
        seg_h = camera_height / num_segments
        lane_segments = {1: [], 2: []}

        if visualize:
            overlay = image.copy()
            overlay[center_mask == 1] = (255, 0, 0)
            overlay[center_mask == 2] = (0, 0, 255)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

        for row in range(num_segments):
            for col in range(num_segments):
                idx = row * num_segments + col
                dist = distances[idx]

                r_start = int(row * seg_h)
                r_end = int((row + 1) * seg_h)
                c_start = int(col * seg_w)
                c_end = int((col + 1) * seg_w)

                cx = int((col + 0.5) * seg_w)
                cy = int((row + 0.5) * seg_h)

                for lane in [1, 2]:
                    if np.any(center_mask[r_start:r_end, c_start:c_end] == lane):
                        lane_segments[lane].append({
                            'row': row, 'col': col,
                            'center_x': cx, 'center_y': cy,
                            'distance': dist
                        })

        target_alphas = {}
        for lane in [1, 2]:
            segs = lane_segments[lane]
            if segs:
                best = min(segs, key=lambda s: abs(s['distance'] - self.pure_pursuit.ld))
                x, y = best['center_x'], best['center_y']
                alpha = ((x - camera_width / 2) / (camera_width / 2)) * (horizontal_fov / 2)
                target_alphas[lane] = alpha

                if visualize:
                    cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
            else:
                target_alphas[lane] = None

        if visualize:
            cv2.imshow("Target Visualization", image)
            cv2.waitKey(1)

        return target_alphas


class PidController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.previous_error = None

    def calculate_speed(self, current_speed, target_speed):
        if np.isnan(current_speed):
            current_speed = 0.0

        error = target_speed - current_speed

        # Reset integral when error changes sign (anti-windup)
        if self.previous_error is not None:
            if error * self.previous_error < 0:
                self.integral = 0.0

        self.integral += error
        derivative = 0.0 if self.previous_error is None else error - self.previous_error
        self.previous_error = error

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


class PurePursuitController:
    def __init__(self, L, min_ld, max_ld, Kdd):
        self.L = L
        self.min_ld = min_ld
        self.max_ld = max_ld
        self.Kdd = Kdd
        self.alpha = None
        self.ld = (min_ld + max_ld) / 2

    def calculate_steering_angle(self, current_speed):
        if self.alpha is None:
            return 0.0

        self.ld = np.clip(self.Kdd * current_speed, self.min_ld, self.max_ld)
        print(self.ld)
        return np.arctan(2 * self.L * np.sin(self.alpha) / self.ld)
