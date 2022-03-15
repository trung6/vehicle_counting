from interface import Detector
import cv2

class VehicleDetector(Detector):
    def __init__(self, accepted_conf):
        super().__init__()
        self.__net = cv2.dnn.readNet('models/mobilenet-ssd/vehicle-detection-adas-0002.xml', 'models/mobilenet-ssd/vehicle-detection-adas-0002.bin')
        self.__accepted_conf = accepted_conf
        # Specify target device
        self.__net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    
    def get_detections(self, frame):
        blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
        self.__net.setInput(blob)
        out = self.__net.forward()
        print(type(out))
        exit()

        # The net outputs a blob with the shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        # for detection in out.reshape(-1, 7):
        #     image_id, label, conf, x1, y1, x2, y2 = detection
        #     if conf > self.__accepted_conf:
if __name__ == '__main__':
    detector = VehicleDetector(0.5)
    detector.get_detections('/media/trung/01D4B61EC8BD72C0/Ki1_nam5/HHTQĐ/TRUNG_vehicle_counting_project/test_image.jpeg')


        



