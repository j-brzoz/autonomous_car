import time
import json
import logging

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from vehicle import Driver #type: ignore
from models import VAE, MLP_classifier, MLP_regressor
from vehicle_control import VehicleController, PidController, PurePursuitController
from distances import prepare_image, show_segments, compute_distances
from obstacle_detection import detect_obstacles, calculate_predicted_distances, calculate_centers_of_segments
from line_segmentation import compute_line_mask, compute_center_mask

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    logging.error("Error: config.json not found.")
    exit()
except json.JSONDecodeError:
    logging.error("Error: config.json is not valid JSON.")
    exit()


# --- Constants ---
TIME_STEP = config.get("time_step", 50)
NUM_THREADS = config.get("num_threads", 8)
OUTPUT_FILE = config.get("output_file", "output.avi")
SIMULATION_LIMIT = config.get("simulation_limit", 1000)
TARGET_FPS = int(1000/TIME_STEP)
MODEL_PATHS = {
    "encoder": config.get("encoder_file_path"),
    "classifier": config.get("classifier_file_path"),
    "left": config.get("left_file_path"),
    "right": config.get("right_file_path"),
    "straight": config.get("straight_file_path"),
    "line_following": config.get("line_following_file_path"),
}
TASKS = list(MODEL_PATHS.keys())
CROP_HEIGHT = 120 # height to crop from the top
RESIZE_SHAPE = (192, 384)
CONSTANT_SPEED_KPH = 5.0
CAMERA_MOUNT_HEIGHT = config.get("camera_mount_height_obstacle")


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
if device.type == "cpu":
    torch.set_num_threads(NUM_THREADS)
    logging.info(f"Number of CPU threads set to: {torch.get_num_threads()}")


# --- Model Loading ---
models = {}
missing_paths = []
try:
    for task in TASKS:
        path = MODEL_PATHS.get(task)
        if not path:
            logging.error(f"No path specified for task '{task}' in config.")
            exit()

        if task == "encoder":
            model = VAE()
        elif task == "classifier":
            model = MLP_classifier()
        else:
            model = MLP_regressor()

        logging.info(f"Loading model for task '{task}' from {path}...")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[task] = model
    
    # Yolop_v2 model for lane segmentation
    path = config.get("yolopv2_file_path")
    yolopv2 = torch.jit.load(path, map_location=device)
    yolopv2.to(device)
    yolopv2.eval()
    models["yolopv2"] = yolopv2

    logging.info("Models loaded successfully!")

except FileNotFoundError as e:
     logging.error(f"Error loading model: {e}. Check file paths in config.json.")
     exit()
except Exception as e:
     logging.error(f"An unexpected error occurred during model loading: {e}")
     exit()


# --- Simulation and Camera Setup ---
try:
	driver = Driver()

	# Devices for driving on intersections
	camera_right = driver.getDevice("camera_right")
	camera_right.enable(TIME_STEP)
	camera_left = driver.getDevice("camera_left")
	camera_left.enable(TIME_STEP)
	camera_front = driver.getDevice("camera_front")
	camera_front.enable(TIME_STEP)

	CAMERA_WIDTH = camera_front.getWidth()
	CAMERA_HEIGHT = camera_front.getHeight()

	# Devices for obstacle avoidance
	camera_right_obstacle = driver.getDevice("camera_right_obstacle")
	camera_right_obstacle.enable(TIME_STEP)
	camera_left_obstacle = driver.getDevice("camera_left_obstacle")
	camera_left_obstacle.enable(TIME_STEP)
	camera_front_obstacle = driver.getDevice("camera_front_obstacle")
	camera_front_obstacle.enable(TIME_STEP)
	lidar_right_obstacle = driver.getDevice("lidar_right_obstacle")
	lidar_right_obstacle.enable(TIME_STEP)
	lidar_left_obstacle = driver.getDevice("lidar_left_obstacle")
	lidar_left_obstacle.enable(TIME_STEP)
	lidar_front_obstacle = driver.getDevice("lidar_front_obstacle")
	lidar_front_obstacle.enable(TIME_STEP)
 
	CAMERA_RIGHT_OBSTACLE_HEIGHT = camera_right_obstacle.getHeight()
	CAMERA_RIGHT_OBSTACLE_WIDTH = camera_right_obstacle.getWidth()
	CAMERA_LEFT_OBSTACLE_HEIGHT = camera_left_obstacle.getHeight()
	CAMERA_LEFT_OBSTACLE_WIDTH = camera_left_obstacle.getWidth()
	CAMERA_FRONT_OBSTACLE_HEIGHT = camera_front_obstacle.getHeight()
	CAMERA_FRONT_OBSTACLE_WIDTH = camera_front_obstacle.getWidth()
	LIDAR_RIGHT_OBSTACLE_HEIGHT = lidar_right_obstacle.getNumberOfLayers()
	LIDAR_RIGHT_OBSTACLE_WIDTH = lidar_right_obstacle.getHorizontalResolution()
	LIDAR_LEFT_OBSTACLE_HEIGHT = lidar_left_obstacle.getNumberOfLayers()
	LIDAR_LEFT_OBSTACLE_WIDTH = lidar_left_obstacle.getHorizontalResolution()
	LIDAR_FRONT_OBSTACLE_HEIGHT = lidar_front_obstacle.getNumberOfLayers()
	LIDAR_FRONT_OBSTACLE_WIDTH = lidar_front_obstacle.getHorizontalResolution()
 
	HORIZONTAL_FOV = lidar_front_obstacle.getFov()
	VERTICAL_FOV = lidar_front_obstacle.getVerticalFov()
 
	logging.info(f"Cameras initialized with resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
except Exception as e:
    logging.error(f"Failed to initialize Webots driver or camera: {e}")
    exit()


# --- Video Writer Setup ---
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, TARGET_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT))
if not out.isOpened():
	logging.warning(f"Could not open video writer for file: {OUTPUT_FILE}")
	exit()
 
# --- Controllers for steering the car ---
pid_speed = PidController(Kp=0.3, Ki=0.055, Kd=0.001)
pid_steering = PidController(Kp=0.45, Ki=0.08, Kd=0.002)
pure_pursuit = PurePursuitController(L=3, min_ld=8, max_ld=15, Kdd=0.5)
vehicle_controller = VehicleController(pure_pursuit, pid_speed, pid_steering, driver, max_speed=30.0, min_speed=12)

# Data for obstacles detection
NUM_SEGMENTS = 20
segments_coordinates = calculate_centers_of_segments(NUM_SEGMENTS, CAMERA_FRONT_OBSTACLE_WIDTH, CAMERA_FRONT_OBSTACLE_HEIGHT)
predicted_distances = calculate_predicted_distances(VERTICAL_FOV, NUM_SEGMENTS, CAMERA_MOUNT_HEIGHT, margin_factor=1)

# --- Image Preprocessing Transform ---
def crop_upper(image: torch.Tensor, crop_height: int = CROP_HEIGHT) -> torch.Tensor:
    """Crops the top portion of the image tensor."""
    if image.shape[1] <= crop_height: # image is C, H, W
        logging.warning(f"Crop height ({crop_height}) is >= image height ({image.shape[1]}). Returning original image.")
        return image
    return image[:, crop_height:, :]

front_transform = v2.Compose([
    v2.ToImage(),                           # convert NumPy HWC to Tensor CHW
    v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale pixels to [0, 1]
    v2.Lambda(lambda img: img[:3, :, :]),   # remove alpha channel if present (make it 3-channel)
    v2.Lambda(lambda img: crop_upper(img, 120)),                  # remove sky (crop top part)
    v2.Resize(RESIZE_SHAPE), # resize to model's expected input size
])

sides_transform = v2.Compose([
    v2.ToImage(),                           # convert NumPy HWC to Tensor CHW
    v2.ToDtype(torch.float32, scale=True),  # convert to float32 and scale pixels to [0, 1]
    v2.Lambda(lambda img: img[:3, :, :]),   # remove alpha channel if present (make it 3-channel)
    v2.Lambda(lambda img: crop_upper(img, 80)),                  # remove sky (crop top part)
    v2.Resize(RESIZE_SHAPE), # resize to model's expected input size
])
logging.info(f"Image transform pipeline created: Crop={CROP_HEIGHT}, Resize={RESIZE_SHAPE}")


# --- Main Loop ---
tick = 0
start_time = time.monotonic()
prev_angle = 0.0

try:
	with torch.no_grad():
		while driver.step() != -1:
			loop_start_time = time.monotonic()

			# 1. Get image and lidar data 
			image_data_front = camera_front.getImage()
			image_data_right = camera_right.getImage()
			image_data_left = camera_left.getImage()
			if not image_data_front or not image_data_right or not image_data_left:
				logging.warning("No image data received from camera.")
				continue
			
			image_data_front_obstacle = prepare_image(
       										camera_front_obstacle.getImage(),
          									CAMERA_FRONT_OBSTACLE_WIDTH,
            								CAMERA_FRONT_OBSTACLE_HEIGHT)
			image_data_left_obstacle = prepare_image(
       										camera_left_obstacle.getImage(),
          									CAMERA_LEFT_OBSTACLE_WIDTH,
            								CAMERA_LEFT_OBSTACLE_HEIGHT)
			image_data_right_obstacle = prepare_image(
       										camera_right_obstacle.getImage(),
          									CAMERA_RIGHT_OBSTACLE_WIDTH,
            								CAMERA_RIGHT_OBSTACLE_HEIGHT)
			lidar_data_front = lidar_front_obstacle.getRangeImage()
			lidar_data_left = lidar_left_obstacle.getRangeImage()
			lidar_data_right = lidar_right_obstacle.getRangeImage()
   
			# 2. Convert to NumPy array (BGRA format from Webots camera)
			camera_image_np_front = np.frombuffer(image_data_front, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
			camera_image_np_right = np.frombuffer(image_data_right, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
			camera_image_np_left = np.frombuffer(image_data_left, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))

			# 3. Convert color BGR for OpenCV display/saving and processing
			frame_for_display_front = cv2.cvtColor(camera_image_np_front, cv2.COLOR_BGRA2BGR)
			frame_for_display_right = cv2.cvtColor(camera_image_np_right, cv2.COLOR_BGRA2BGR)
			frame_for_display_left = cv2.cvtColor(camera_image_np_left, cv2.COLOR_BGRA2BGR)

			# 4. Preprocess image for NN input (directly from BGR numpy array)
			img_tensor_front = front_transform(frame_for_display_front).to(device)
			img_tensor_front = img_tensor_front.unsqueeze(0) # add batch dimension, shape becomes [1, C, H, W]
			img_tensor_right = sides_transform(frame_for_display_right).to(device)
			img_tensor_right = img_tensor_right.unsqueeze(0) # add batch dimension, shape becomes [1, C, H, W]
			img_tensor_left = sides_transform(frame_for_display_left).to(device)
			img_tensor_left = img_tensor_left.unsqueeze(0) # add batch dimension, shape becomes [1, C, H, W]

			# 5. Model Inference
			mu_front, logvar_front = models["encoder"].encode(img_tensor_front)
			latent_front = models["encoder"].reparameterize(mu_front, logvar_front)
			mu_right, logvar_right = models["encoder"].encode(img_tensor_right)
			latent_right = models["encoder"].reparameterize(mu_right, logvar_right)
			mu_left, logvar_left = models["encoder"].encode(img_tensor_left)
			latent_left = models["encoder"].reparameterize(mu_left, logvar_left)
					
			latent = torch.cat((latent_front, latent_left, latent_right), dim=1)
					
			probs = models["classifier"](latent) # shape might be [1, 1] or [1]

			is_line_following = probs.item() > 0.5 

			# 6. Determine Steering Angle
			angle = 0.0
			navigation_task = "right"
			if is_line_following or vehicle_controller.is_passing:
				# Detect obstacle
				obstacle_front = detect_obstacles(
					image_data_front_obstacle, CAMERA_FRONT_OBSTACLE_HEIGHT, CAMERA_FRONT_OBSTACLE_WIDTH,
					LIDAR_FRONT_OBSTACLE_WIDTH, LIDAR_FRONT_OBSTACLE_HEIGHT, lidar_data_front,
					segments_coordinates, predicted_distances, 15, HORIZONTAL_FOV, 6, NUM_SEGMENTS, display_results=False)
				obstacle_left = detect_obstacles(
					image_data_left_obstacle, CAMERA_LEFT_OBSTACLE_HEIGHT, CAMERA_LEFT_OBSTACLE_WIDTH,
					LIDAR_LEFT_OBSTACLE_WIDTH, LIDAR_LEFT_OBSTACLE_HEIGHT, lidar_data_left,
					segments_coordinates, predicted_distances, 5, HORIZONTAL_FOV, 10, NUM_SEGMENTS, display_results=False)
				obstacle_right = detect_obstacles(
					image_data_right_obstacle, CAMERA_RIGHT_OBSTACLE_HEIGHT, CAMERA_RIGHT_OBSTACLE_WIDTH,
					LIDAR_RIGHT_OBSTACLE_WIDTH, LIDAR_RIGHT_OBSTACLE_HEIGHT, lidar_data_right,
					segments_coordinates, predicted_distances, 5, HORIZONTAL_FOV, 10, NUM_SEGMENTS, display_results=False)
    
				obstacles = {"front": obstacle_front, "left": obstacle_left, "right": obstacle_right}
    
				# Make line segmentation with yolop and compute centers of the lines
				line_mask = compute_line_mask(
        			image_data_front_obstacle, CAMERA_FRONT_OBSTACLE_HEIGHT,
           			CAMERA_FRONT_OBSTACLE_WIDTH, models["yolopv2"], visualize=False)
				center_mask = compute_center_mask(line_mask)
				distances_front = compute_distances(segments_coordinates, CAMERA_FRONT_OBSTACLE_HEIGHT,
                    CAMERA_FRONT_OBSTACLE_WIDTH, LIDAR_FRONT_OBSTACLE_WIDTH, LIDAR_FRONT_OBSTACLE_HEIGHT, lidar_data_front)
    
				# Update vehicle controller
				angle = vehicle_controller.get_steering_for_line_following(
        			distances_front, center_mask, CAMERA_FRONT_OBSTACLE_WIDTH, CAMERA_FRONT_OBSTACLE_HEIGHT,
           			NUM_SEGMENTS, HORIZONTAL_FOV, obstacles, image_data_front_obstacle, visualize=False)
				vehicle_controller.update_passing_status(obstacles)
				navigation_mode = "Line Following"
			else:
				# placeholder for navigation logic - 'left', 'right', or 'straight'		
				navigation_mode = navigation_task.capitalize()
				angle_tensor = models[navigation_task](latent)
				angle = angle_tensor.item()

			# 7. Control Vehicle
            # increase progression
			if navigation_task == "left":
				angle = max(min(angle*2.5, 0.5), -0.5)
			elif navigation_task == "right":
				angle = max(min(angle*2.5, 0.7), -0.7)
					
			target_speed = vehicle_controller.get_target_speed()
			vehicle_controller.update(angle, target_speed)

			# 8. Display & Save Video Frame (using the original size BGR frame)
			cv2.imshow("Camera Feed", frame_for_display_front)
			if out.isOpened():
				out.write(frame_for_display_front)

			# allow window events to be processed
			if cv2.waitKey(1) & 0xFF == ord('q'):
				logging.info("Quit signal received.")
				break

			# 9. Logging
			tick += 1
			loop_time = time.monotonic() - loop_start_time
			real_angle = driver.getSteeringAngle()
			real_speed = driver.getCurrentSpeed()
			logging.info(f"Tick: {tick}, Angle: {real_angle:.4f}, Speed: {real_speed:.4f}, Loop Time: {loop_time:.4f}s, Mode: {navigation_mode}, Probs {probs}, Is passing: {vehicle_controller.is_passing}")


			# 10. Check simulation limit
			# if tick >= SIMULATION_LIMIT:
			# 	logging.info(f"Simulation limit ({SIMULATION_LIMIT} ticks) reached.")
			# 	break

finally:
    # --- Cleanup ---
    logging.info("Cleaning up resources...")
    if 'out' in locals() and out.isOpened():
        out.release()
        logging.info("Video writer released.")
    cv2.destroyAllWindows()
    logging.info("OpenCV windows destroyed.")

    end_time = time.monotonic()
    total_time = end_time - start_time
    logging.info(f"Simulation finished. Total time: {total_time:.2f}s, Ticks: {tick}")
    if tick > 0:
        logging.info(f"Average FPS: {tick / total_time:.2f}")

print("Exited")