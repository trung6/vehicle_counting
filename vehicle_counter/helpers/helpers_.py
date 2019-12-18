import numpy as np

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