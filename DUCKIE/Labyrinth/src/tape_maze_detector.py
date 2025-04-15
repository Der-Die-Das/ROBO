import cv2
import numpy as np
import math
import os # Needed for path operations if debugging output is added later
# Optional: Import necessary for type hinting if desired
from typing import Tuple, List, Optional, Dict, Any

# --- Attempt to import extended image processing for skeletonization ---
XIMGPROC_AVAILABLE = False
try:
    import cv2.ximgproc
    XIMGPROC_AVAILABLE = True
    print("cv2.ximgproc found, skeletonization will be used.")
except ImportError:
    print("Warning: cv2.ximgproc not found. Skeletonization unavailable. Line detection might be less robust.")
    # Consider alternative strategies if skeletonization is critical and unavailable


class TapeMazeDetector:
    """
    Detects junctions and available cardinal directions from a BGR image
    containing paths (e.g., colored tape).
    Internally processes a derived mask from the image.
    """
    def __init__(self,
                 current_heading: str = "north",
                 border_crop_pixels: int = 5,
                 angle_tolerance: float = 25.0,
                 forward_top_threshold_px: int = 50,
                 mask_crop_percentage: float = 30.0,
                 mask_lower_threshold: int = 130,
                 # --- Additional parameters inferred from typical processing ---
                 hough_rho: float = 1.0,                # Distance resolution of the accumulator in pixels.
                 hough_theta: float = np.pi / 180.0,  # Angle resolution of the accumulator in radians.
                 hough_threshold: int = 20,          # Accumulator threshold parameter. Only lines > threshold are returned.
                 hough_min_line_length: int = 20,    # Minimum line length. Line segments shorter than this are rejected.
                 hough_max_line_gap: int = 10):      # Maximum allowed gap between points on the same line to link them.
        """
        Initialize the detector with robot's current heading and detection parameters.

        Args:
            current_heading (str): Initial heading ("north", "east", "south", "west").
            border_crop_pixels (int): Pixels to ignore near the mask border when analyzing lines.
            angle_tolerance (float): Tolerance in degrees for classifying line angles into cardinal directions.
            forward_top_threshold_px (int): How close (in pixels) a vertical line's top endpoint
                                           must be to the top edge of the mask to be considered 'forward'.
            mask_crop_percentage (float): Percentage of image height (from bottom) to use for mask generation.
            mask_lower_threshold (int): Lower bound for color channel values (BGR) for mask generation.
            hough_rho (float): HoughLinesP parameter: Distance resolution.
            hough_theta (float): HoughLinesP parameter: Angle resolution (radians).
            hough_threshold (int): HoughLinesP parameter: Accumulator threshold.
            hough_min_line_length (int): HoughLinesP parameter: Minimum line length.
            hough_max_line_gap (int): HoughLinesP parameter: Maximum gap between line segments.
        """
        # --- Original Attributes ---
        self.current_heading: str = current_heading
        self.cardinal_map: Dict[str, int] = {"north": 0, "east": 90, "south": 180, "west": 270}
        self.inv_cardinal_map: Dict[int, str] = {v: k for k, v in self.cardinal_map.items()}
        # Relative angles (relative to robot's heading): North=0, East=90, South=180, West=270
        # We map these to image angles later.

        # --- New Attributes from Batch Script Logic & Defaults ---
        self.border_crop_pixels: int = border_crop_pixels
        self.angle_tolerance: float = angle_tolerance
        self.forward_top_threshold_px: int = forward_top_threshold_px
        self.mask_crop_percentage: float = mask_crop_percentage
        self.mask_lower_threshold: int = mask_lower_threshold

        # --- Hough Transform Parameters ---
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

        # --- Input Validation ---
        if current_heading not in self.cardinal_map:
            raise ValueError("Initial heading must be one of: north, east, south, west.")
        if not (0 <= mask_crop_percentage <= 100):
            raise ValueError("mask_crop_percentage must be between 0 and 100.")
        if not (0 <= mask_lower_threshold <= 255):
             raise ValueError("mask_lower_threshold must be between 0 and 255.")
        if not (0 <= angle_tolerance < 90):
            raise ValueError("angle_tolerance must be between 0 and 90 degrees.")
        if border_crop_pixels < 0:
            raise ValueError("border_crop_pixels cannot be negative.")
        if forward_top_threshold_px < 0:
             raise ValueError("forward_top_threshold_px cannot be negative.")

        # Placeholder for debug data if needed
        self.debug_data: Dict[str, Optional[np.ndarray]] = {}

    def update_heading(self, new_heading: str):
        """
        Update the detector's current heading state.

        Args:
            new_heading (str): The new heading ("north", "east", "south", "west").

        Raises:
            ValueError: If the new_heading is invalid.
        """
        if new_heading in self.cardinal_map:
            self.current_heading = new_heading
            # print(f"Detector heading updated to: {self.current_heading}") # Optional debug
        else:
            raise ValueError("Heading must be one of: north, east, south, west.")

    def _calculate_angle_diff(self, angle1: float, angle2: float) -> float:
        """Calculates the minimum difference between two angles (0-360 degrees)."""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    def _map_angle_to_cardinal(self, angle_deg: float, y1: int, y2: int, mask_height: int) -> Optional[str]:
        """
        Maps a detected line angle (relative to the image frame) to a *relative*
        cardinal direction (relative to the image frame: 90=Up/Forward, 0=Right,
        180=Left, 270=Down/Back) based on angle tolerance and forward threshold.

        Args:
            angle_deg (float): The angle of the line (0=horizontal right, 90=vertical up).
            y1, y2 (int): Y-coordinates of the line endpoints.
            mask_height (int): Height of the mask image where the line was detected.

        Returns:
            Optional[str]: The corresponding *relative* direction ("forward", "right", "left", "back")
                           or None if it doesn't match criteria.
        """
        top_y = min(y1, y2)

        # Special Check: Forward Direction (Vertical line near top)
        # Angle needs to be near vertical (90 degrees in image coords)
        if self._calculate_angle_diff(angle_deg, 90.0) <= self.angle_tolerance:
             # AND top endpoint must be close to the top edge of the mask
             if top_y <= self.forward_top_threshold_px:
                 # print(f"  DEBUG: Forward line detected (Angle: {angle_deg:.1f}, TopY: {top_y})")
                 return "forward" # Relative forward

        # Check: Right Direction (Horizontal line near 0/360 degrees)
        if self._calculate_angle_diff(angle_deg, 0.0) <= self.angle_tolerance:
            # print(f"  DEBUG: Right line detected (Angle: {angle_deg:.1f})")
            return "right" # Relative right

        # Check: Left Direction (Horizontal line near 180 degrees)
        if self._calculate_angle_diff(angle_deg, 180.0) <= self.angle_tolerance:
            # print(f"  DEBUG: Left line detected (Angle: {angle_deg:.1f})")
            return "left" # Relative left

        # Check: Backward Direction (Vertical line near 270 degrees, NOT near top)
        # Usually filtered out, but included for completeness
        # if self._calculate_angle_diff(angle_deg, 270.0) <= self.angle_tolerance:
        #     # Ensure it's not the 'forward' line misclassified
        #     if not (self._calculate_angle_diff(angle_deg, 90.0) <= self.angle_tolerance and top_y <= self.forward_top_threshold_px):
        #         # print(f"  DEBUG: Back line detected (Angle: {angle_deg:.1f})")
        #         return "back" # Relative back (usually the path we came from)

        # print(f"  DEBUG: Angle {angle_deg:.1f} did not map to a cardinal direction.")
        return None

    def _get_absolute_direction(self, relative_direction: str) -> Optional[str]:
        """ Converts a relative direction ('forward', 'left', 'right') to an absolute cardinal direction."""
        current_angle = self.cardinal_map[self.current_heading]
        relative_map = {"forward": 0, "right": 90, "left": 270, "back": 180} # Relative angle offsets

        if relative_direction not in relative_map:
            return None

        relative_offset = relative_map[relative_direction]
        # Note: Cardinal map uses North=0, East=90. Image angles use Right=0, Up=90.
        # Let's align: Robot North (0) maps to Image Up (90). Robot East (90) -> Image Right (0).
        # Robot South (180) -> Image Down (270). Robot West (270) -> Image Left (180).
        # To convert relative robot dir to absolute cardinal dir:
        # target_angle_absolute = (current_angle + relative_offset) % 360

        # Example: current='north'(0), relative='forward'(0) -> target=0 ('north')
        # Example: current='north'(0), relative='right'(90) -> target=90 ('east')
        # Example: current='north'(0), relative='left'(270) -> target=270 ('west')
        # Example: current='east'(90), relative='forward'(0) -> target=90 ('east')
        # Example: current='east'(90), relative='right'(90) -> target=180 ('south')
        # Example: current='east'(90), relative='left'(270) -> target=360%360=0 ('north')

        absolute_angle = (current_angle + relative_offset) % 360
        return self.inv_cardinal_map.get(absolute_angle)


    def process_image(self, frame: np.ndarray, return_debug: bool = False) -> Any: # Type hint allows Tuple or complex dict
        """
        Processes an input BGR image 'frame' to detect junctions and available directions.

        Internally performs:
        1. Cropping & Mask Generation
        2. Mask Cleaning
        3. Skeletonization (if available)
        4. Line Detection & Filtering
        5. Direction Analysis & Junction Detection

        Args:
            frame (np.ndarray): The input image in BGR format.
            return_debug (bool): If True, returns a dictionary with intermediate steps.
                                 If False (default), returns Tuple[bool, List[str]].

        Returns:
            Union[Tuple[bool, List[str]], Dict[str, Any]]:
                - Default: (junction_detected, directions)
                - If return_debug=True: A dictionary containing 'junction_detected',
                  'directions', and intermediate images ('mask_raw', 'mask_cleaned',
                  'skeleton', 'skel_edges', 'lines_filtered_vis').
        """
        self.debug_data = {} # Reset debug data

        # --- 1. Crop & Generate Mask ---
        height, width, _ = frame.shape
        crop_height_pixels = int(height * (self.mask_crop_percentage / 100.0))
        start_row = max(0, height - crop_height_pixels)
        lower_region_bgr = frame[start_row:height, :]

        mask_raw = None
        if lower_region_bgr.shape[0] > 0 and lower_region_bgr.shape[1] > 0:
            lower_bgr = np.array([self.mask_lower_threshold] * 3)
            upper_bgr = np.array([255] * 3)
            mask_raw = cv2.inRange(lower_region_bgr, lower_bgr, upper_bgr)
            self.debug_data['mask_raw'] = mask_raw
        else:
             print("Warning: Cropped region for mask has zero size.")
             if return_debug:
                self.debug_data.update({
                    'mask_raw': None, 'mask_cleaned': None, 'skeleton': None,
                    'skel_edges': None, 'lines_filtered_vis': None,
                    'junction_detected': False, 'directions': []
                })
                return self.debug_data
             else:
                return False, [] # No mask -> no detection

        if mask_raw is None or np.sum(mask_raw) == 0:
            print("Warning: Mask generation failed or resulted in an empty mask.")
            if return_debug:
                self.debug_data.update({
                    'mask_raw': mask_raw, 'mask_cleaned': None, 'skeleton': None,
                    'skel_edges': None, 'lines_filtered_vis': None,
                    'junction_detected': False, 'directions': []
                })
                return self.debug_data
            else:
                return False, []

        mask_height, mask_width = mask_raw.shape

        # --- 2. Mask Cleaning ---
        kernel = np.ones((3,3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.debug_data['mask_cleaned'] = mask_cleaned

        # --- 3. Skeletonization (Optional) ---
        skeleton = None
        skel_edges = None
        target_for_hough = mask_cleaned # Default to cleaned mask edges if skeleton fails

        if XIMGPROC_AVAILABLE:
            # Ensure mask is binary (0 or 255)
            _, binary_mask = cv2.threshold(mask_cleaned, 127, 255, cv2.THRESH_BINARY)
            skeleton = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            self.debug_data['skeleton'] = skeleton
            # Use Canny on skeleton for cleaner lines
            skel_edges = cv2.Canny(skeleton, 50, 150)
            self.debug_data['skel_edges'] = skel_edges
            if np.sum(skel_edges) > 0: # Check if Canny found anything
                 target_for_hough = skel_edges
            else:
                 print("Warning: Skeletonization or Canny edge detection on skeleton yielded empty result. Falling back to mask edges.")
                 target_for_hough = cv2.Canny(mask_cleaned, 50, 150) # Fallback
        else:
            # Fallback: Use Canny directly on the cleaned mask
            print("Info: Skeletonization unavailable. Using Canny on cleaned mask for line detection.")
            target_for_hough = cv2.Canny(mask_cleaned, 50, 150)
            self.debug_data['skel_edges'] = target_for_hough # Store Canny edges here

        # --- 4. Line Detection ---
        lines = cv2.HoughLinesP(target_for_hough,
                                self.hough_rho,
                                self.hough_theta,
                                self.hough_threshold,
                                minLineLength=self.hough_min_line_length,
                                maxLineGap=self.hough_max_line_gap)

        # --- 5. Line Filtering & Angle Analysis ---
        filtered_lines = []
        relative_directions_found = set() # Store relative directions ('forward', 'left', 'right')

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Filter lines touching the border
                if (x1 <= self.border_crop_pixels or x1 >= mask_width - self.border_crop_pixels or
                    y1 <= self.border_crop_pixels or y1 >= mask_height - self.border_crop_pixels or
                    x2 <= self.border_crop_pixels or x2 >= mask_width - self.border_crop_pixels or
                    y2 <= self.border_crop_pixels or y2 >= mask_height - self.border_crop_pixels):
                    continue # Skip line touching border

                # Calculate angle (adjusting for coordinate system: 0=right, 90=up)
                dx = x2 - x1
                dy = y1 - y2 # Invert dy because image y increases downwards
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)
                if angle_deg < 0:
                    angle_deg += 360 # Normalize to 0-360

                # Map angle to relative direction
                relative_dir = self._map_angle_to_cardinal(angle_deg, y1, y2, mask_height)

                if relative_dir:
                    relative_directions_found.add(relative_dir)
                    filtered_lines.append(line) # Keep line if it mapped to a direction

        # --- 6. Convert Relative to Absolute Directions ---
        absolute_directions = set()
        for rel_dir in relative_directions_found:
            abs_dir = self._get_absolute_direction(rel_dir)
            if abs_dir:
                absolute_directions.add(abs_dir)

        # --- 7. Exclude Incoming Direction ---
        # The direction the robot *came from* is the opposite of its current heading.
        current_angle = self.cardinal_map[self.current_heading]
        incoming_angle = (current_angle + 180) % 360
        incoming_direction = self.inv_cardinal_map.get(incoming_angle)

        if incoming_direction in absolute_directions:
             # print(f"  DEBUG: Removing incoming direction '{incoming_direction}'")
             absolute_directions.remove(incoming_direction)

        # Convert set to sorted list for consistent output
        final_directions = sorted(list(absolute_directions), key=lambda d: self.cardinal_map[d])

        # --- 8. Junction Detection ---
        # Simple heuristic: More than one available path means a junction
        # (Could be refined by contour analysis, line proximity etc. if needed)
        junction_detected = len(final_directions) > 1

        # DEBUG

        raw_filename = "debug_raw_input_frame.png"
        final_processed_filename = "debug_final_hough_input.png"

        # Save raw frame (make sure frame is not None)
        if frame is not None:
            cv2.imwrite(raw_filename, frame)

        # Save the image used for Hough line detection (target_for_hough)
        # Check if it exists and is not empty
        if target_for_hough is not None and np.sum(target_for_hough) > 0:
            cv2.imwrite(final_processed_filename, target_for_hough)

        ## END DEBUG

        # --- 9. Prepare Debug Output (if requested) ---
        if return_debug:
            lines_vis = cv2.cvtColor(target_for_hough, cv2.COLOR_GRAY2BGR) # Start with edges/skeleton
            if filtered_lines:
                 for line in filtered_lines:
                     cv2.line(lines_vis, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2) # Green
            self.debug_data['lines_filtered_vis'] = lines_vis
            self.debug_data['junction_detected'] = junction_detected
            self.debug_data['directions'] = final_directions
            # Add contours if needed:
            # contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # self.debug_data['contours'] = contours # Store raw contours
            return self.debug_data
        else:
            return junction_detected, final_directions