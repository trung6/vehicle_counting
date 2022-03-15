# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
from datetime import date, datetime

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# Object detection imports
# import sys
# from os import path
# sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import helpers.label_map_util as label_map_util
from helpers.helpers_ import remove_box_with_score_thres, nms, filter_classes, load_image_into_numpy_array

from tracker.sort import Sort, iou
from detector.interface import Detector

interested_classes = np.array([3, 6, 8])
roi_relative_with_cameras = [0.8, 0.8, 0.6]

MODEL_NAME = 'models/ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('models', 'ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
# uncomment if you have not download the model yet

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, 
# so that when our convolution network predicts 5, we know that this corresponds to airplane.
# Here I use internal utility functions, 
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class VehicleDetector(Detector):
    def __init__(self):
        super().__init__()
    
    def get_detections(self, frame):
        """
        Args:
        - image: a 3-d int Tensor of shape [img_h, img_w, 3]

        Returns:
        - boxes: a 3d float Tensor of shape [1, num_boxes, 4] \
            representing predicted bounding boxes relative with image size and in form of [y1, x1, y2, x2]
        - scores: a 2d float Tensor of shape [1, num_boxes] representing scores for predicted bounding boxes which are float reals between 0 and 1.
        - classes: a 2d float Tensor of shape [1, num_boxes] representing class ids of objects in boxes.
        - num: a scalar representing the number of predicted bounding boxes.
        """
        img_h, img_w, _ = frame.shape
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                image_np_expanded = np.expand_dims(frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                                detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})

                deleted_indices = remove_box_with_score_thres(boxes[0], scores[0], score_thres=0.5)
                
                ##delete 
                boxes = np.delete(boxes[0], deleted_indices, axis=0).astype(float)
                scores = np.delete(scores[0], deleted_indices, axis=0).astype(float)
                classes = np.delete(classes[0], deleted_indices, axis=0).astype(int)

                selected_indices = filter_classes(classes, interested_classes)
                
                ##gather
                boxes = boxes[selected_indices, :][0]
                scores = scores[selected_indices][0]
                classes = classes[selected_indices][0]

                trans = np.array([img_h, img_w, img_h, img_w])

                ##convert from relative lengths into absolute lengths
                boxes = np.dot(boxes, np.diag(trans))

                ##convert from [y1, x1, y2, x2] into [x1, y1, x2, y2]
                boxes[:, [0, 1]] = boxes[:, [1, 0]]
                boxes[:, [2, 3]] = boxes[:, [3, 2]]

                ##perform nms.
                selected_indices = nms(boxes, scores, iou_threshold=0.5)
                
                ##gather remaining bounding boxes, scores, classes after nms. 
                boxes = boxes[ selected_indices , : ] #tf.gather(boxes, selected_indices)
                scores = scores[ selected_indices ]
                classes = classes[ selected_indices ]
                # print(boxes.shape, scores.shape, classes.shape, num.shape, boxes)
                # exit()
                ##convert scores in form of [num_boxes, 1]
                scores = np.expand_dims(scores, axis=1)  #np.expand_dims(scores[0], axis=1)
                
                ##gather boxes and scores into array in form of [x1, y1, x2, y2, score]
                bboxes_with_scores = np.append(boxes, scores, axis=1)

        return bboxes_with_scores
if __name__ == '__main__':
    detector = VehicleDetector()
    frame = cv2.imread('/media/trung/01D4B61EC8BD72C0/Ki1_nam5/HHTQĐ/TRUNG_vehicle_counting_project/vehicle_counter/test_data/test_image.jpeg')
    detections = detector.get_detections(frame)
    print(detections)
