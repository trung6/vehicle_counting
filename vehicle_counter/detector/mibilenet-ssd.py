import cv2



net = cv2.dnn.readNet('models/mobilenet-ssd/vehicle-detection-adas-0002.xml', 'models/mobilenet-ssd/vehicle-detection-adas-0002.bin')

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
