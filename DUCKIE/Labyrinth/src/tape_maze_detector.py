import cv2
import numpy as np
import math

class TapeMazeDetector:
    def __init__(self, current_heading="north"):
        """
        Initialize with a starting heading.
        """
        self.current_heading = current_heading  # "north", "east", "south", or "west"
        self.cardinal_map = {"north": 0, "east": 90, "south": 180, "west": 270}
        self.inv_cardinal_map = {v: k for k, v in self.cardinal_map.items()}

    def update_heading(self, new_heading):
        if new_heading in self.cardinal_map:
            self.current_heading = new_heading
        else:
            raise ValueError("Heading must be one of: north, east, south, west.")

    def _map_angle_to_cardinal(self, angle):
        offset = self.cardinal_map[self.current_heading]
        global_angle = (angle + offset) % 360
        cardinal_angle = round(global_angle / 90) * 90 % 360
        return self.inv_cardinal_map[cardinal_angle]

    def process_image(self, frame):
        """
        Process an image (BGR) and detect a junction and available cardinal directions.
        Returns:
          junction_detected (bool), directions (list of strings)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        junction_detected = False
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        if len(large_contours) >= 2:
            junction_detected = True
        elif large_contours:
            x, y, w, h = cv2.boundingRect(large_contours[0])
            if w > frame.shape[1] * 0.5 or h > frame.shape[0] * 0.5:
                junction_detected = True

        edges = cv2.Canny(red_mask, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        detected_cardinals = set()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = math.atan2((y2-y1), (x2-x1))
                angle_deg = math.degrees(angle_rad) % 360
                cardinal = self._map_angle_to_cardinal(angle_deg)
                detected_cardinals.add(cardinal)
        directions = list(detected_cardinals)
        return junction_detected, directions