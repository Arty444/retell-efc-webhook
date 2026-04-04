"""Acoustic SLAM: Sound Localization And Mapping via phone movement.

As the user walks while holding the phone, the IMU (accelerometer + gyroscope)
tracks position changes via pedestrian dead reckoning. Each position provides
a new angle-of-arrival measurement to the bird. Multiple measurements from
different positions create a triangulation problem that yields distance.

This is geometric parallax — the same principle astronomers use to measure
star distances, applied to acoustics at human scale.

Math:
  Position A: measure angle θ_A to sound source
  Walk baseline b to Position B: measure angle θ_B
  Distance d = b * sin(θ_A) * sin(θ_B) / sin(θ_A - θ_B)

With continuous measurements and a Kalman filter, we refine the estimate
as the user naturally moves.
"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IMUReading:
    """A single IMU measurement from the phone."""
    timestamp: float          # seconds (unix epoch)
    accel_x: float = 0.0     # m/s² in phone frame
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0      # rad/s
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    heading: float = 0.0     # compass degrees
    pitch: float = 90.0      # device pitch
    roll: float = 0.0        # device roll


@dataclass
class PositionAngleMeasurement:
    """An angle measurement from a known position."""
    timestamp: float
    x: float                  # position in meters (local frame, start = 0,0)
    y: float
    azimuth: float            # measured angle to bird (degrees, compass)
    elevation: float          # measured elevation angle (degrees)
    confidence: float         # confidence of angle measurement


class PedestrianDeadReckoning:
    """
    Track user position from IMU data using step detection.

    Uses accelerometer magnitude peaks to detect steps, compass heading
    for direction, and a configurable step length model.

    More accurate than double-integrating accelerometer (which drifts badly).
    Step detection + heading is the standard approach for indoor positioning.
    """

    def __init__(self, step_length: float = 0.7):
        """
        Args:
            step_length: Average step length in meters (adjustable per user height)
        """
        self.step_length = step_length
        self.x = 0.0
        self.y = 0.0
        self.total_distance = 0.0
        self.step_count = 0

        # Step detection state
        self._accel_buffer: list[tuple[float, float]] = []  # (timestamp, magnitude)
        self._last_step_time = 0.0
        self._step_cooldown = 0.3  # minimum seconds between steps
        self._step_threshold = 1.2  # acceleration magnitude threshold (g-units)

        # Heading filter
        self._heading_buffer: list[float] = []

    def update(self, imu: IMUReading) -> Optional[tuple[float, float]]:
        """
        Process an IMU reading. Returns (x, y) position if a step was detected.
        """
        # Compute acceleration magnitude (in g)
        accel_mag = math.sqrt(imu.accel_x**2 + imu.accel_y**2 + imu.accel_z**2) / 9.81
        self._accel_buffer.append((imu.timestamp, accel_mag))

        # Keep 1 second of history
        cutoff = imu.timestamp - 1.0
        self._accel_buffer = [(t, a) for t, a in self._accel_buffer if t > cutoff]

        # Track heading
        self._heading_buffer.append(imu.heading)
        if len(self._heading_buffer) > 10:
            self._heading_buffer.pop(0)

        # Step detection: peak in acceleration magnitude
        if (len(self._accel_buffer) >= 3 and
                imu.timestamp - self._last_step_time > self._step_cooldown):

            # Check if current sample is a local maximum above threshold
            recent = [a for _, a in self._accel_buffer[-5:]]
            if len(recent) >= 3:
                mid = len(recent) // 2
                if (recent[mid] > self._step_threshold and
                        recent[mid] >= max(recent[mid-1], recent[min(mid+1, len(recent)-1)])):

                    self._last_step_time = imu.timestamp
                    self.step_count += 1

                    # Average heading over recent samples (circular mean)
                    heading_rad = math.radians(self._circular_mean(self._heading_buffer))

                    # Update position
                    dx = self.step_length * math.sin(heading_rad)
                    dy = self.step_length * math.cos(heading_rad)
                    self.x += dx
                    self.y += dy
                    self.total_distance += self.step_length

                    return (self.x, self.y)

        return None

    def get_position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def _circular_mean(self, angles_deg: list[float]) -> float:
        if not angles_deg:
            return 0.0
        sin_sum = sum(math.sin(math.radians(a)) for a in angles_deg)
        cos_sum = sum(math.cos(math.radians(a)) for a in angles_deg)
        return math.degrees(math.atan2(sin_sum, cos_sum)) % 360


class AcousticSLAM:
    """
    Fuse IMU-tracked positions with angle-of-arrival measurements
    to triangulate the bird's position.

    Collects (position, angle) pairs as the user walks. With 2+ measurements
    from different positions, triangulates distance. With 3+, uses least-squares
    for robust estimation.

    The key insight: even casual walking creates enough baseline for distance
    estimation. Walking 2 meters sideways while a bird is 20 meters away
    creates a 5.7° parallax angle — well within our 4-mic array's resolution.
    """

    def __init__(self):
        self.pdr = PedestrianDeadReckoning()
        self.measurements: list[PositionAngleMeasurement] = []
        self.max_measurements = 30  # keep recent measurements
        self.estimated_position: Optional[tuple[float, float, float]] = None  # x, y, z of bird

    def add_imu_reading(self, imu: IMUReading) -> Optional[tuple[float, float]]:
        """Process IMU data and return new position if step detected."""
        return self.pdr.update(imu)

    def add_angle_measurement(
        self,
        azimuth: float,
        elevation: float,
        confidence: float,
    ):
        """Record an angle measurement at the current position."""
        x, y = self.pdr.get_position()
        self.measurements.append(PositionAngleMeasurement(
            timestamp=time.time(),
            x=x, y=y,
            azimuth=azimuth,
            elevation=elevation,
            confidence=confidence,
        ))

        # Trim old measurements
        if len(self.measurements) > self.max_measurements:
            self.measurements = self.measurements[-self.max_measurements:]

    def estimate_distance(self) -> Optional[dict]:
        """
        Triangulate bird position from accumulated measurements.

        Requires at least 2 measurements from different positions
        (minimum ~0.5m apart for meaningful parallax).

        Returns:
            {
                "distance": float,         # meters from current position
                "bird_x": float,           # estimated bird position
                "bird_y": float,
                "bird_z": float,           # estimated height (from elevation angles)
                "confidence": float,
                "baseline": float,         # total baseline used
                "n_measurements": int,
                "method": "acoustic_slam",
            }
        """
        if len(self.measurements) < 2:
            return None

        # Check that we have sufficient spatial diversity (baseline)
        positions = [(m.x, m.y) for m in self.measurements]
        max_baseline = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                d = math.sqrt((positions[i][0] - positions[j][0])**2 +
                              (positions[i][1] - positions[j][1])**2)
                max_baseline = max(max_baseline, d)

        if max_baseline < 0.3:  # Less than 30cm baseline — not enough parallax
            return None

        # Triangulation via least-squares line intersection
        # Each measurement defines a ray from position (x,y) in direction azimuth
        # Find the point that minimizes distance to all rays

        def _ray_cost(bird_pos):
            """Sum of squared distances from bird_pos to each measurement ray."""
            bx, by = bird_pos
            total = 0.0
            for m in self.measurements:
                # Ray direction from measurement position
                az_rad = math.radians(m.azimuth)
                dx = math.sin(az_rad)
                dy = math.cos(az_rad)

                # Vector from measurement position to candidate bird position
                vx = bx - m.x
                vy = by - m.y

                # Perpendicular distance from point to ray
                # d = |v × d| / |d| = |vx*dy - vy*dx|
                perp_dist = abs(vx * dy - vy * dx)

                # Only count if the bird is in front of the ray (not behind)
                along = vx * dx + vy * dy
                if along < 0:
                    perp_dist += abs(along) * 2  # heavy penalty for behind

                total += perp_dist**2 * m.confidence

            return total

        # Grid search for initial estimate
        best_cost = float("inf")
        best_pos = (0.0, 0.0)
        current_x, current_y = self.pdr.get_position()

        # Search in a 100m radius
        for dx in range(-50, 51, 5):
            for dy in range(-50, 51, 5):
                candidate = (current_x + dx, current_y + dy)
                c = _ray_cost(candidate)
                if c < best_cost:
                    best_cost = c
                    best_pos = candidate

        # Refine with Nelder-Mead
        from scipy.optimize import minimize
        result = minimize(_ray_cost, best_pos, method="Nelder-Mead",
                          options={"xatol": 0.1, "maxiter": 200})
        bird_x, bird_y = result.x

        # Distance from current position
        distance = math.sqrt((bird_x - current_x)**2 + (bird_y - current_y)**2)

        # Estimate height from elevation angles
        elevations = [m.elevation for m in self.measurements if abs(m.elevation) > 1]
        if elevations:
            avg_elevation = np.mean(elevations)
            bird_z = distance * math.tan(math.radians(avg_elevation))
        else:
            bird_z = 0.0

        # Confidence based on baseline, measurement count, and residual
        baseline_conf = min(1.0, max_baseline / 3.0)  # 3m baseline = full confidence
        count_conf = min(1.0, len(self.measurements) / 5.0)
        residual_conf = max(0, 1.0 - result.fun / (len(self.measurements) + 1))
        confidence = (baseline_conf * 0.4 + count_conf * 0.3 + residual_conf * 0.3)

        self.estimated_position = (bird_x, bird_y, bird_z)

        return {
            "distance": round(distance, 1),
            "bird_x": round(bird_x, 1),
            "bird_y": round(bird_y, 1),
            "bird_z": round(bird_z, 1),
            "confidence": round(min(1.0, confidence), 3),
            "baseline": round(max_baseline, 2),
            "n_measurements": len(self.measurements),
            "method": "acoustic_slam",
        }

    def reset(self):
        """Reset SLAM state for a new target."""
        self.pdr = PedestrianDeadReckoning()
        self.measurements = []
        self.estimated_position = None
