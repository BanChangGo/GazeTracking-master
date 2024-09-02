import collections
import numpy as np
import time

class EyeMovementTracker:
    def __init__(self, window_size=25, horizontal_threshold=(0.56, 0.77), max_pupil_shift=5):
        self.window_size = window_size
        self.horizontal_ratios = collections.deque(maxlen=window_size)
        self.pupil_positions = collections.deque(maxlen=window_size)
        self.horizontal_threshold = horizontal_threshold
        self.max_pupil_shift = max_pupil_shift
        self.last_update_time = None 

    #큐에 정보 저장해서 update, 수평 이동인지 검사 이후 종료
    def update(self, horizontal_ratio, pupil_coords):
        current_time = time.time()
        # Debug: Print the incoming horizontal_ratio and pupil_coords
        print(f"Update called with horizontal_ratio: {horizontal_ratio}, pupil_coords: {pupil_coords}")

        if self.last_update_time is not None:
            time_diff = current_time - self.last_update_time
            print(f"Time since last update: {time_diff:.4f} seconds")  # 이전 업데이트와의 시간 차이 출력
        self.last_update_time = current_time


        if horizontal_ratio is None:
            print("Skipping update because horizontal_ratio is None.")
            return
        
        
        # 수평 비율과 pupil 위치를 큐에 저장
        self.horizontal_ratios.append(horizontal_ratio)
        self.pupil_positions.append(pupil_coords)
        
        # Debug: Print the current state of the queues
        print("Current horizontal_ratios queue:", list(self.horizontal_ratios))
        print("Current pupil_positions queue:", list(self.pupil_positions))

        # 수평 이동을 검증
        if len(self.horizontal_ratios) == self.horizontal_ratios.maxlen:
            if self._is_horizontal_movement() and self._are_pupil_positions_within_bounds():
                print("Horizontal movement detected!")
                ##face_detectro 실행을 
                # 이동 탐지 후 초기화
                self.horizontal_ratios.clear()
                self.pupil_positions.clear()
                self.initial_pupil_pos = None
                print("Exiting the program due to detected horizontal movement.")
                exit()  # 프로그램 종료
    
    # 수평 이동 비율 중 min, max 인덱스 값 return
    def return_index_of(self):
        """Check if the horizontal ratio moved from min to max within the queue."""
        min_index = self.horizontal_ratios.index(min(self.horizontal_ratios))
        max_index = self.horizontal_ratios.index(max(self.horizontal_ratios))
        minMax = [min_index, max_index] #max값은 left, min값은 right이 되어야 함
        return minMax

    # 좌, 우 이동 검사 
    def _is_horizontal_movement(self):
        minMax = self.return_index_of()

        if minMax[0] < minMax[1]: #왼쪽보다 오른쪽을 먼저봤다면 false로 처리
            return False
         
        return (self.horizontal_ratios[minMax[0]] < self.horizontal_threshold[0] and
                self.horizontal_ratios[minMax[1]] > self.horizontal_threshold[1])

    #좌표 변환 없는지 검사 
    def _is_within_threshold(self, current_pupil_pos):
        """Check if the pupil's vertical movement is within the allowed shift threshold."""
        return np.abs(current_pupil_pos[1] - self.pupil_positions[0][1]) < self.max_pupil_shift

    def _are_pupil_positions_within_bounds(self):
        """Check if the pupil positions between min and max indices are within the allowed shift."""
        minMax = self.return_index_of()
        min_index = minMax[0] #오른쪽 봤을 때의 que의 index
        max_index = minMax[1]

        # Extract pupil positions between min and max indices
        for i in range(max_index, min_index + 1):
            if not self._is_within_threshold(self.pupil_positions[i]):
                return False
        return True