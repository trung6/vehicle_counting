import torch
from __init__ import VehicleCounter

import datetime
from tracker.sort import Sort


class Controller(VehicleCounter):
    def __init__(self, total_passed_vehicle = 0, ids_met_roi = [], write_now = False):
        super().VehicleCounter()
        self.__total_passed_vehicle = total_passed_vehicle #private
        self.__ids_met_roi = ids_met_roi
        self.__write_now = write_now

    def count_on_frame(self, frame):
        tracker = Sort(max_age=25, min_hits=3)
        
    
    def check_time(self):
        pass




        
        