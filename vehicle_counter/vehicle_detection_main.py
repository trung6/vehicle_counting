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
from utils import label_map_util
from sort import Sort, iou

#helper function
def remove_box_with_score_thres(boxes, scores, score_thres):
    """
    Args:
    - boxes: A 2-D float Tensor of shape [num_boxes, 4].
    - scores: A 1-D float Tensor of shape [num_boxes] representing a single score corresponding to each box (each row of boxes).
    - score_thres: A scalar float Tensor representing threshold to remove a box from input boxes.
    
    Returns:
    - selected_boxes: A 2-D float Tensor of shape [num_boxes, 4] representing boxes after selection.
    - indices: a int tuple representing indices to delete. 
    """
    indices = np.where(scores<score_thres)
    return indices

def nms(dets, scores, iou_threshold):
    """
    Args:
    - dets: a 2-D float Tensor of shape [num_boxes, 4] representing predicted bounding boxes in form of [x1, y1, x2, y2]
    - scores: a 1-D float Tensor of shape [num_boxes] representing confidence scores for bounding boxes.
    - iou_threshold: a float scalar representing iou threshold to remove boxes that coincide.
    Returns:
    - keep: a int list representing indices of dets that are kept after nms.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # print(x1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]#sort in decreasing

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def filter_classes(labels, interested_classes):
    """
    Args:
    - labels: a 1-D int numpy array of shape (num_boxes) representing class_ids of predicted bounding boxes
    - interested_classes: a 1-D int numpy array of shape (num_interest_classes) representing class_ids of interested_classes.
    Returns
    - indices: a list of int scalar of len (num_boxes) representing keeping indices of labels. 
    """
    return np.array(np.where(np.in1d(labels, interested_classes)))

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)

# initialize .csv
# print(os.path.isdir('/media/trung/01D4B61EC8BD72C0/Ki1_nam5/HHTQĐ/vehicle_counting_tensorflow-master/traffic_measurement.csv'))
# if not os.path.isdir('/media/trung/01D4B61EC8BD72C0/Ki1_nam5/HHTQĐ/vehicle_counting_tensorflow-master/traffic_measurement.csv'):
#     with open('traffic_measurement.csv', 'w') as f:
#         writer = csv.writer(f)
#         csv_line = \
#             'Camera_id, Start time, End time, Num of vehicles'
#         writer.writerows([csv_line.split(',')])

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

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

def detector(image):
    """
    Args:
    - image: a 3-d int Tensor of shape [img_h, img_w, 3]

    Returns:
    - boxes: a 3d float Tensor of shape [1, num_boxes, 4] representing predicted bounding boxes relative with image size and in form of [y1, x1, y2, x2]
    - scores: a 2d float Tensor of shape [1, num_boxes] representing scores for predicted bounding boxes which are float reals between 0 and 1.
    - classes: a 2d float Tensor of shape [1, num_boxes] representing class ids of objects in boxes.
    - num: a scalar representing the number of predicted bounding boxes.
    """
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

            image_np_expanded = np.expand_dims(image, axis=0)

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

# Detection
@tf.function
def object_detection_function():
    total_passed_vehicle = 0 #count the number of vehicles
    
    #Constants
    interested_classes = np.array([3, 6, 8])
    tracker = Sort(max_age=15, min_hits=0)
    roi_relative_with_cameras = [0.8, 0.8, 0.6]
    
    # input video
    # cap = cv2.VideoCapture('cam1.mp4')
    cam_id = 2
    video_name = 'cam{}.mp4'.format(cam_id)
    cap = cv2.VideoCapture(video_name)
    cam_name = video_name.split('.')[0]

    _, img1 = cap.read()
    h = img1.shape[0]
    w = img1.shape[1]
    video_writer = cv2.VideoWriter('demo_cam{}.mp4'.format(cam_id), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (w, h))
    write_in_video = True

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            summary_writer = tf.summary.FileWriter('graphs', sess.graph)

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            ##track objects that met roi line
            last_id_met_roi_line = -1
            ids_met_roi = []
            
            # for all the frames that are extracted from input video
            while cap.isOpened():
                
                draw_roi_line = True

                (ret, input_frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                img_h, img_w, _ = input_frame.shape
                ROI_POSITION = int(roi_relative_with_cameras[cam_id - 1] * img_h)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

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
                
                ##perform tracker.
                trackers = tracker.update(bboxes_with_scores)##trackers is a 2D float Tensor of shape [num_boxes, 5] in form of [x1, y1, x2, y2, id]
                
                ##report last id that met roi line.
                
                ####check_logic(trackers, ids_met_roi, frame) 
                for track in trackers:
                    if (track[3] > ROI_POSITION) and (int(track[4]) not in ids_met_roi): #(int(track[4]) != int(last_id_met_roi_line)):
                        total_passed_vehicle += 1 
                        last_id_met_roi_line = int(track[4])
                        ids_met_roi.append(int(track[4]))
                        print('id met roi line: {}'.format(str(last_id_met_roi_line)))
                        cv2.line(input_frame, (0, ROI_POSITION), (img_w, ROI_POSITION), (0, 0xFF, 0), 5)
                        x1, y1, x2, y2 = track[:4]
                        cv2.rectangle(input_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
                        draw_roi_line = False
                        
                    # else: 
                    #     cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)
                if draw_roi_line:
                    cv2.line(input_frame, (0, ROI_POSITION), (img_w, ROI_POSITION), (0, 0, 0xFF), 5)                

                ##insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Total vehicles passed ROI line: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                if write_in_video:
                    video_writer.write(input_frame)
                    
                cv2.imshow('image',input_frame)
                cv2.waitKey(1)
            #end while
            with open('traffic_measurement.csv', 'a') as f:
                writer_csv = csv.writer(f)
                
                today = date.today().strftime('%d/%m/%Y')
                start_time = int(datetime.now().strftime('%H'))
                end_time = start_time + 1

                csv_line = '{},{},{},{},{}'.format(cam_name, today,start_time, end_time, total_passed_vehicle)
                writer_csv.writerows([csv_line.split(',')])

            cap.release()
            cv2.destroyAllWindows()
        summary_writer.close()
##end function

if __name__=='__main__':
    object_detection_function()		
