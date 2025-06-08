import cv2
import numpy as np
import torch


def compute_line_mask(image, camera_height, camera_width, model, visualize=False):
    """
    Performs lane line segmentation on a raw camera image using a YOLOP model.

    Args:
        image (np.ndarray): Raw image in BGRA format.
        camera_height (int): Height of the original camera image.
        camera_width (int): Width of the original camera image.
        model (torch.jit.ScriptModule): Pre-trained YOLOP model for lane detection.
        visualize (bool): Whether to visualize the result. Default is False.

    Returns:
        np.ndarray: Rescaled binary lane mask (2D array).
    """
    # Resize to model input resolution
    img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    # Prepare image for model
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().to('cpu') / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    img = img.to(next(model.parameters()).device)

    # Run inference
    [_, _], _, lane_mask = model(img)

    # Post-processing
    lane_mask = torch.round(lane_mask).squeeze(1).int().squeeze().cpu().numpy()
    lane_mask_rescaled = cv2.resize(
        lane_mask.astype(np.uint8),
        (camera_width, camera_height),
        interpolation=cv2.INTER_NEAREST
    )

    if visualize:
        # Optional: convert to color for easier visualization
        mask_colored = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        mask_colored[lane_mask_rescaled == 1] = (255, 0, 0)  # Blue color for lane
        cv2.imshow("Lane Mask (Rescaled)", mask_colored)
        cv2.waitKey(1)

    return lane_mask_rescaled


def compute_center_mask(binary_mask, d_min=30, d_max=300, window_h=10, thickness=3):
    """
    Computes a center mask with two lane traces based on the largest lane segment.

    Args:
        binary_mask (np.ndarray): Binary lane mask (2D array).
        d_min (int): Minimum lane offset.
        d_max (int): Maximum lane offset.
        window_h (int): Height of the sliding window.
        thickness (int): Thickness of the drawn traces.

    Returns:
        np.ndarray: Binary mask with the drawn lane traces.
    """
    h, w = binary_mask.shape

    # Step 1: Find the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:
        return np.zeros_like(binary_mask, np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    points = np.argwhere(labels == largest_label)

    # Step 2: Compute average x-coordinates in sliding windows
    points = points[np.argsort(points[:, 0])]
    centers = []
    y_start = None

    for start in range(0, h, window_h):
        segment = points[(points[:, 0] >= start) & (points[:, 0] < start + window_h)]
        if segment.size == 0:
            continue

        y_mean = float(segment[:, 0].mean())
        x_mean = float(segment[:, 1].mean())
        centers.append((y_mean, x_mean))

        if y_start is None:
            y_start = y_mean

    if not centers:
        return np.zeros_like(binary_mask, np.uint8)

    ys_cent, xs_cent = zip(*centers)
    ys_cent = np.array(ys_cent, dtype=np.float32)
    xs_cent = np.array(xs_cent, dtype=np.float32)

    # Step 3: Interpolate x-coordinates for all y in [y_start, h-1]
    full_ys = np.arange(int(np.ceil(y_start)), h, dtype=np.float32)
    x_interp = np.interp(full_ys, ys_cent, xs_cent)

    # Step 4: Draw traces with offset
    center_mask = np.zeros((h, w), np.uint8)

    for idx, y in enumerate(full_ys.astype(np.int32)):
        x_center = x_interp[idx]
        alpha = (y - y_start) / float(h - y_start)
        d = d_min + alpha * (d_max - d_min)

        x_left = int(round(x_center + d))
        x_right = int(round(x_center - d))

        if 0 <= x_left < w:
            cv2.circle(center_mask, (x_left, y), thickness, 2, -1)
        if 0 <= x_right < w:
            cv2.circle(center_mask, (x_right, y), thickness, 1, -1)

    return center_mask
