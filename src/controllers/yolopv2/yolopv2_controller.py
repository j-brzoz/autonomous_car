import time
from vehicle import Driver
import cv2
import numpy as np
import torch
import json
from utils import letterbox, driving_area_mask, lane_line_mask,\
 split_for_trace_model, non_max_suppression, plot_one_box, scale_coords, clip_coords


# config file
with open("config.json", "r") as f:
    config = json.load(f)


# set timestep
TIME_STEP = config["time_step"]


# import model
model_file_path = config["model_file_path"]
model = torch.jit.load(model_file_path, map_location='cpu')
model.to('cpu')
model.eval()

torch.set_num_threads(config["num_threads"]) 
print(f"Num threads: {torch.get_num_threads()}")
print("Got the model!")


# from highway overtake, needs a rewrite
def apply_PID(position, targetPosition):
    """Apply the PID controller and return the angle command."""
	# needs tuning
    p_coefficient = 0.4
    i_coefficient = 0.000015
    d_coefficient = 0.01
    diff = position - targetPosition
    if apply_PID.previousDiff is None:
        apply_PID.previousDiff = diff
    # anti-windup mechanism
    if diff > 0 and apply_PID.previousDiff < 0:
        apply_PID.integral = 0
    if diff < 0 and apply_PID.previousDiff > 0:
        apply_PID.integral = 0
    apply_PID.integral += diff
    # compute angle
    angle = p_coefficient * diff + i_coefficient * apply_PID.integral + d_coefficient * (diff - apply_PID.previousDiff)
    apply_PID.previousDiff = diff
    return angle

apply_PID.integral = 0
apply_PID.previousDiff = None


# get devices
driver = Driver()
camera = driver.getDevice("camera")
camera.enable(TIME_STEP)
CAMERA_WIDTH = camera.getWidth()
CAMERA_HEIGHT = camera.getHeight()
tracking_interval = 4 # perform traking on every tracking_interval tick
tick = 0


# uncomment those lines to save the video
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter(config["output_file"], fourcc, 20.0, (640,480))



def detect_lane_centers(mask):
	"""Naive lane detection."""
	if isinstance(mask, torch.Tensor):
		mask = mask.squeeze().cpu().numpy()

	# mask is a camera image with lanes marked as 1 and everything else as 0
	if np.sum(mask) > 0:                             # if any lane detected
		lanes = np.mean(mask[-50:], 0)               # consider only bottom 50px
		left_lane = lanes[:mask.shape[1] // 2]       # find the left lane
		if np.sum(left_lane) > 0:
			# compute weighted sum (bottom pixels matter more)
			left_lane = np.sum(left_lane * range(0, len(left_lane))) / np.sum(left_lane)
			return int(left_lane)
	return None


# where we want to have the left lane (in px)
target = 135

loop_time = time.monotonic()
with torch.no_grad():
	while driver.step() != -1:
		# get image from camera and convert to useful format
		image_data = camera.getImage()
		camera_image_np = np.frombuffer(image_data, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
		cv2_image = cv2.cvtColor(camera_image_np, cv2.COLOR_BGRA2BGR)
		# 640x480 is the required format for yolopv2
		img = cv2.resize(cv2_image, (640,480), interpolation=cv2.INTER_NEAREST)
		output = img.copy()

		# helper flag
		got_lane = False

		if tick % tracking_interval == 0:
			# image conversion
			img = img.transpose(2, 0, 1)
			img = torch.from_numpy(img).to('cpu')
			img = img.float()
			img /= 255.0
			if img.ndimension() == 3:
				img = img.unsqueeze(0)

			# main computaion cost

			# for object detection, road segmentation, lane detection
			[pred, anchor_grid], segmentation, lanes = model(img)

			# flags for visualization
			masking = True
			obj_det = False

			# PID input (where we are) 
			direction = target 

			if masking:
				# prepare segmentation mask
				seg_mask = segmentation
				_, seg_mask = torch.max(seg_mask, 1)
				seg_mask = seg_mask.int().squeeze().cpu().numpy()

				# prepare lane mask
				lane_mask = lanes
				lane_mask = torch.round(lane_mask).squeeze(1)
				lane_mask = lane_mask.int().squeeze().cpu().numpy()

				# lane detection
				lane = detect_lane_centers(lane_mask)

				# color segmentation(green) and lanes(red)
				color_area = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
				color_area[seg_mask == 1] = [0, 255, 0]
				color_area[lane_mask == 1] = [255, 0, 0]

				# convert color area from RGB to BGR format (for OpenCV)
				color_area = color_area[..., ::-1]

				# compute a grayscale mask from the colored area
				color_mask = np.mean(color_area, 2)

				# blend the original output image with the color_area
				output[color_mask != 0] = output[color_mask != 0] * 0.5 + color_area[color_mask != 0] * 0.5

				# if lane found
				if lane:
					# put lane marker (yellow)
					output[-50:, lane-1:lane+1] = [0, 255, 255]
					direction = lane
					got_lane = True
				else:
					# if no lane found, go straight (direction = target)
					direction = target
					got_lane = False

				# put target marker (blue)
				output[-50:, target-1:target+1] = [255, 0, 0]

			# for object detection, not use currently
			if obj_det:		
				pred = split_for_trace_model(pred,anchor_grid)
				pred = non_max_suppression(pred)
				pred0 = pred[0]

				img0_shape = output.shape
				clip_coords(pred0, img0_shape)

				for det in pred0:
					*xyxy, _, _ = det
					plot_one_box(xyxy, output)

			cv2.imshow("YOLOPv2", output)
			cv2.waitKey(10)

		# uncomment those lines to save the video
		# out.write(output)

		tick += 1

		# set constant speed (kph)
		speed = 50
		driver.setCruisingSpeed(speed)
        
		# apply pid
		pid_angle = apply_PID(direction/100, target/100)
		angle = max(min(pid_angle, 0.5), -0.5)

		# print logs
		print(f"Tick: {tick}, Current: {direction}, Target: {target}, Angle: {angle}")
		
		#  angle is set in radians, 
		#  a positive angle steers right and a negative angle steers left
		#  if no lanes drive straight
		if got_lane:
			driver.setSteeringAngle(angle)
		else:
			driver.setSteeringAngle(0)

		# end simulation
		if tick == config["simulation_limit"]:
			print("Stopping the simulation")
			break

# uncomment those lines to save the video 
# out.release()

cv2.destroyAllWindows()

print("Exited")