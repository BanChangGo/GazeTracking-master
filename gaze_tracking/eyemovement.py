import collections
import numpy as np

class EyeMovementTracker:
    def __init__(self, window_size=15, horizontal_threshold=(0.56, 0.77), max_pupil_shift=5):
        self.window_size = window_size
        self.horizontal_ratios = collections.deque(maxlen=window_size)
        self.pupil_positions = collections.deque(maxlen=window_size)
        self.initial_pupil_pos = None
        self.horizontal_threshold = horizontal_threshold
        self.max_pupil_shift = max_pupil_shift

    def update(self, horizontal_ratio, pupil_coords):
        # Debug: Print the incoming horizontal_ratio and pupil_coords
        print(f"Update called with horizontal_ratio: {horizontal_ratio}, pupil_coords: {pupil_coords}")

        if horizontal_ratio is None:
            print("Skipping update because horizontal_ratio is None.")
            return
        
        # 초기 pupil 위치를 저장
        if self.initial_pupil_pos is None:
            self.initial_pupil_pos = pupil_coords
        
        # 수평 비율과 pupil 위치를 큐에 저장
        self.horizontal_ratios.append(horizontal_ratio)
        self.pupil_positions.append(pupil_coords)
        
        # Debug: Print the current state of the queues
        print("Current horizontal_ratios queue:", list(self.horizontal_ratios))
        print("Current pupil_positions queue:", list(self.pupil_positions))

        # 수평 이동을 검증
        if len(self.horizontal_ratios) == self.horizontal_ratios.maxlen:
            if self._is_horizontal_movement() and self._is_pupil_within_bounds():
                print("Horizontal movement detected!")
                # 이동 탐지 후 초기화
                self.horizontal_ratios.clear()
                self.pupil_positions.clear()
                self.initial_pupil_pos = None

    def _is_horizontal_movement(self):
        """Check if the horizontal ratio has moved from left to right within the specified range."""
        return (self.horizontal_ratios[0] < self.horizontal_threshold[0] and
                self.horizontal_ratios[-1] > self.horizontal_threshold[1])

    def _is_pupil_within_bounds(self):
        """Check if the pupil position has not moved significantly in the vertical direction."""
        return np.abs(self.pupil_positions[-1][1] - self.initial_pupil_pos[1]) < self.max_pupil_shift
