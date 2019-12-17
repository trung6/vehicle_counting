from abc import ABC, abstractmethod

##Interface
class Detector(ABC):
    
    @abstractmethod
    def get_detections(self, image):
        pass
