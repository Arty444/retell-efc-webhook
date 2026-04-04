"""Camera-based bird detection for visual rangefinding.

Uses the phone's camera feed to detect birds via simple motion/color
detection heuristics (no ML model needed on-device). The detected
bounding box is fed to distance_estimator.estimate_distance_visual()
for size-based rangefinding.

For production use, this could be replaced with a lightweight YOLO or
MobileNet-SSD bird detector. The heuristic approach works surprisingly
well for birds in flight or perched against sky/foliage backgrounds.
"""

import numpy as np
from typing import Optional
import base64
import logging

logger = logging.getLogger(__name__)


def detect_bird_in_frame(
    image_data: bytes,
    image_width: int,
    image_height: int,
    target_heading: Optional[float] = None,
    camera_heading: Optional[float] = None,
    camera_fov: float = 67.0,
) -> Optional[dict]:
    """
    Detect a bird-like object in a camera frame.

    Uses a combination of:
    1. Motion detection (frame differencing)
    2. Small-object detection (connected components of appropriate size)
    3. Direction prior (if we know where the bird should be from audio)

    Args:
        image_data: Raw image bytes (RGB or grayscale)
        image_width: Frame width in pixels
        image_height: Frame height in pixels
        target_heading: Expected bird heading from audio (degrees)
        camera_heading: Current camera/phone heading (degrees)
        camera_fov: Camera horizontal field of view (degrees)

    Returns:
        {
            "bbox": (x, y, width, height),  # pixel coordinates
            "center_x": int,
            "center_y": int,
            "pixel_size": int,              # max dimension in pixels
            "confidence": float,
            "in_frame": bool,               # whether bird is likely in camera view
            "expected_x": float,            # where in frame bird should be (0-1)
        }
    """
    result = {
        "bbox": None,
        "confidence": 0.0,
        "in_frame": False,
        "expected_x": 0.5,
    }

    # If we know the audio direction, compute where in the frame the bird should be
    if target_heading is not None and camera_heading is not None:
        relative_angle = target_heading - camera_heading
        while relative_angle > 180:
            relative_angle -= 360
        while relative_angle < -180:
            relative_angle += 360

        # Is the bird within the camera's field of view?
        if abs(relative_angle) <= camera_fov / 2:
            result["in_frame"] = True
            # Map angle to horizontal pixel position (0 = left, 1 = right)
            result["expected_x"] = 0.5 + (relative_angle / camera_fov)
        else:
            result["in_frame"] = False
            result["expected_x"] = 0.0 if relative_angle < 0 else 1.0

    # Try to detect bird-like objects in the frame
    try:
        frame = _decode_frame(image_data, image_width, image_height)
        if frame is None:
            return result

        bbox = _find_bird_candidate(frame, image_width, image_height,
                                     result.get("expected_x", 0.5),
                                     result.get("in_frame", False))
        if bbox:
            x, y, w, h = bbox
            result["bbox"] = bbox
            result["center_x"] = x + w // 2
            result["center_y"] = y + h // 2
            result["pixel_size"] = max(w, h)

            # Confidence based on size plausibility and direction agreement
            size_ratio = max(w, h) / max(image_width, image_height)
            # Birds typically occupy 0.5% to 15% of the frame
            if 0.005 <= size_ratio <= 0.15:
                result["confidence"] = 0.5
            elif 0.002 <= size_ratio <= 0.25:
                result["confidence"] = 0.3
            else:
                result["confidence"] = 0.1

            # Boost confidence if detection aligns with audio direction
            if result["in_frame"] and result["expected_x"] is not None:
                detected_x = (x + w / 2) / image_width
                x_error = abs(detected_x - result["expected_x"])
                if x_error < 0.15:
                    result["confidence"] = min(0.8, result["confidence"] + 0.25)
                elif x_error < 0.3:
                    result["confidence"] = min(0.7, result["confidence"] + 0.1)

    except Exception as e:
        logger.debug("Visual detection failed: %s", e)

    return result


# Frame differencing state for motion detection
_prev_frame = None


def _decode_frame(
    image_data: bytes,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Decode raw image bytes to a grayscale numpy array."""
    try:
        arr = np.frombuffer(image_data, dtype=np.uint8)

        # Try as grayscale
        if len(arr) == width * height:
            return arr.reshape((height, width))

        # Try as RGB
        if len(arr) == width * height * 3:
            rgb = arr.reshape((height, width, 3))
            # Convert to grayscale: 0.299R + 0.587G + 0.114B
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
            return gray

        # Try as RGBA
        if len(arr) == width * height * 4:
            rgba = arr.reshape((height, width, 4))
            gray = (0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2]).astype(np.uint8)
            return gray

        return None
    except Exception:
        return None


def _find_bird_candidate(
    gray: np.ndarray,
    width: int,
    height: int,
    expected_x: float,
    has_direction_prior: bool,
) -> Optional[tuple[int, int, int, int]]:
    """
    Find the most bird-like object in a grayscale frame.

    Uses contrast-based detection: birds are typically darker or lighter
    than their background (sky, foliage). We look for small, compact
    high-contrast regions.
    """
    global _prev_frame

    # Downsample for speed (process at 1/4 resolution)
    scale = 4
    small_h, small_w = height // scale, width // scale
    if small_h < 10 or small_w < 10:
        return None

    small = gray[::scale, ::scale][:small_h, :small_w]

    candidates = []

    # Method A: Motion detection via frame differencing
    if _prev_frame is not None and _prev_frame.shape == small.shape:
        diff = np.abs(small.astype(np.int16) - _prev_frame.astype(np.int16)).astype(np.uint8)
        motion_thresh = 25
        motion_mask = diff > motion_thresh

        if np.any(motion_mask):
            bbox = _mask_to_bbox(motion_mask, scale)
            if bbox and _is_bird_sized(bbox, width, height):
                candidates.append(("motion", bbox, 0.4))

    _prev_frame = small.copy()

    # Method B: Dark objects against bright background (birds against sky)
    mean_val = np.mean(small)
    if mean_val > 140:  # Bright background (sky-like)
        dark_mask = small < (mean_val - 50)
        if np.any(dark_mask):
            bbox = _mask_to_bbox(dark_mask, scale)
            if bbox and _is_bird_sized(bbox, width, height):
                candidates.append(("dark_on_bright", bbox, 0.3))

    # Method C: High-contrast edges (any background)
    # Simple edge detection via horizontal + vertical gradients
    if small.shape[0] > 2 and small.shape[1] > 2:
        gx = np.abs(small[:, 1:].astype(np.int16) - small[:, :-1].astype(np.int16))
        gy = np.abs(small[1:, :].astype(np.int16) - small[:-1, :].astype(np.int16))
        # Pad to match dimensions
        gx = np.pad(gx, ((0, 0), (0, 1)), mode='edge')
        gy = np.pad(gy, ((0, 1), (0, 0)), mode='edge')
        edges = np.minimum(gx[:small_h, :small_w] + gy[:small_h, :small_w], 255).astype(np.uint8)

        edge_thresh = 40
        edge_mask = edges > edge_thresh
        if np.any(edge_mask):
            bbox = _mask_to_bbox(edge_mask, scale)
            if bbox and _is_bird_sized(bbox, width, height):
                candidates.append(("edges", bbox, 0.2))

    if not candidates:
        return None

    # Score candidates: prefer ones near expected_x if we have direction info
    best = None
    best_score = -1
    for method, bbox, base_score in candidates:
        x, y, w, h = bbox
        center_x_norm = (x + w / 2) / width
        score = base_score

        if has_direction_prior:
            x_error = abs(center_x_norm - expected_x)
            score += max(0, 0.3 - x_error)  # Bonus for matching expected position

        # Prefer compact (roughly square) objects — birds aren't super elongated
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < 3:
            score += 0.1

        if score > best_score:
            best_score = score
            best = bbox

    return best


def _mask_to_bbox(
    mask: np.ndarray,
    scale: int,
) -> Optional[tuple[int, int, int, int]]:
    """Convert a boolean mask to a bounding box, scaled back to original resolution."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    x = int(cmin * scale)
    y = int(rmin * scale)
    w = int((cmax - cmin + 1) * scale)
    h = int((rmax - rmin + 1) * scale)

    return (x, y, w, h)


def _is_bird_sized(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> bool:
    """Check if a bounding box is a plausible bird size."""
    _, _, w, h = bbox
    max_dim = max(w, h)
    frame_max = max(frame_w, frame_h)

    # Bird should be between ~0.2% and ~25% of the frame
    ratio = max_dim / frame_max
    return 0.002 <= ratio <= 0.25


def reset_motion_state():
    """Reset frame differencing state (e.g., when camera moves significantly)."""
    global _prev_frame
    _prev_frame = None
