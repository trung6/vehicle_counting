from abc import ABC, abstractmethod
import sort.py
##interface
class Tracker(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_trackers(self, detections):
        pass

