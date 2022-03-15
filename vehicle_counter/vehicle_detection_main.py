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


# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.


def detector(image):
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
    
            cap.release()
            cv2.destroyAllWindows()
        summary_writer.close()
##end function

if __name__=='__main__':
    object_detection_function()		
