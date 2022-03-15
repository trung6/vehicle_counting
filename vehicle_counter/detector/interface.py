from abc import ABC, abstractmethod

##Interface
class Detector(ABC):
    
    @abstractmethod
    def get_detections(self, image):
        """
        Getting detections
        Args: 
        image: 3D int Tensor of shape (img_h, img_w, 3) representing an image.
        Returns:
        detections: 2D float Tensor of shape [num_boxes, 5] which each row is in form of [x1, y1, x2, y2, score] \
            representing predicted bounding boxes with scores
        """
        pass