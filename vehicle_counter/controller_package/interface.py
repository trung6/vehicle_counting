from abc import ABC, abstractmethod

##Interface

class VehicleCounter(ABC):
    def __init__(self, total_passed_vehicles=0):
        super().__init__()
        self.__total_passed_vehicles = total_passed_vehicles

    @abstractmethod
    def count_on_frame(self, frame):
        pass

    @abstractmethod
    def check_time(self):
        pass
