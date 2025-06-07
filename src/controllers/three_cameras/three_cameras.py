import time
import json
import logging

import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from vehicle import Driver 
from models import VAE, MLP_classifier, MLP_regressor

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

	camera_right = driver.getDevice("camera_right")
	camera_right.enable(TIME_STEP)
	camera_left = driver.getDevice("camera_right")
	camera_left.enable(TIME_STEP)
	camera_front = driver.getDevice("camera_front")
	camera_front.enable(TIME_STEP)

	CAMERA_WIDTH = camera_front.getWidth()
	CAMERA_HEIGHT = camera_front.getHeight()
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
no_line_follow_counter = 0
MAX_BRAKE_INTENSITY = 1.0
BRAKE_SCALING = 0.2 
try:
	with torch.no_grad():
		while driver.step() != -1:
			loop_start_time = time.monotonic()

			# 1. Get Image
			image_data_front = camera_front.getImage()
			image_data_right = camera_right.getImage()
			image_data_left = camera_left.getImage()
			if not image_data_front or not image_data_right or not image_data_left:
				logging.warning("No image data received from camera.")
				continue

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

			is_line_following = probs.item() > 0.50

			# 6. Determine Steering Angle
			angle = 0.0
			navigation_task = "right"
			if is_line_following:
				angle_tensor = models["line_following"](latent)
				angle = angle_tensor.item()
				navigation_mode = "Line Following"
			else:
				# placeholder for navigation logic - 'left', 'right', or 'straight'		
				navigation_mode = navigation_task.capitalize()
				angle_tensor = models[navigation_task](latent)
				angle = angle_tensor.item()

			# 7. Control Vehicle
			# scale angle
			if navigation_task == "left":
				max_angle = 0.5
			elif navigation_task == "right":
				max_angle = 0.7
			angle = max(min(angle*1.5, max_angle), (-1)*max_angle)         
			
			# clamp angle change to +- 0.1
			max_angle_change = 0.1
			angle = max(min(angle, prev_angle + max_angle_change), prev_angle - max_angle_change)
			driver.setSteeringAngle(angle)
			prev_angle = angle
			driver.setSteeringAngle(angle)
			
			# calculate speed based on angle
			speed = 8 + 30 * ((max_angle-np.abs(angle))/max_angle)**4
                  
			# calculate if need to brake
			if not is_line_following:
				no_line_follow_counter += 1
				no_line_follow_counter = min(5, no_line_follow_counter)
			else:
				no_line_follow_counter = 0
			brake_intensity = min(MAX_BRAKE_INTENSITY, BRAKE_SCALING * no_line_follow_counter)
			if no_line_follow_counter == 0 or driver.getCurrentSpeed() < 8:
				driver.setCruisingSpeed(speed)
				driver.setBrakeIntensity(0)
			else:
				driver.setCruisingSpeed(0)
				driver.setBrakeIntensity(brake_intensity)
			
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
			real_break_intensity = driver.getBrakeIntensity()
			logging.info(f"Tick: {tick}, Speed: {real_speed:.4f}, Break: {real_break_intensity:.4f}, Angle: {real_angle:.4f}, Loop Time: {loop_time:.4f}s, Mode: {navigation_mode}")


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