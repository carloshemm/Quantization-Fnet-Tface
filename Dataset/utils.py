from ast import Return
from urllib import response
import cv2
import numpy as np
from numpy import clip
import torch
import torchvision
import torch.nn.functional as F
import json
import configparser


def resize_with_padding(image, size=(224,224)):
    '''
    Resizes a black and white image to the specified size, 
    adding padding to preserve the aspect ratio.
    '''
    padded_image = np.zeros(size, dtype=np.uint8)
    try:
        
        # Get the height and width of the image
        height, width = image.shape[:2]
        # Calculate the aspect ratio of the image
        aspect_ratio = height / width
        # Calculate the new height and width after resizing to (224,224)
        new_height, new_width = size
        if aspect_ratio > 1:
            new_width = int(new_height / aspect_ratio)
        else:
            new_height = int(new_width * aspect_ratio)
        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_NEAREST)
        # Create a black image with the target size
        # Calculate the number of rows/columns to add as padding
        padding_rows = (size[0] - new_height) // 2
        padding_cols = (size[1] - new_width) // 2
        # Add the resized image to the padded image, with padding on the left and right sides
        padded_image = cv2.copyMakeBorder(resized_image,padding_rows,padding_rows,padding_cols,padding_cols, cv2.BORDER_CONSTANT, None, value = 0)
        padded_image = cv2.resize(padded_image, size, interpolation = cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error in resize_with_padding: {e}")
        padded_image = np.ones((size[0],size[1],3), dtype=np.uint8) * 255
    
    return padded_image


def cut_roi(frame, roi):
    p1 = roi.position.astype(int)
    p1 = clip(p1, [0, 0], [frame.shape[-1], frame.shape[-2]])
    p2 = (roi.position + roi.size).astype(int)
    p2 = clip(p2, [0, 0], [frame.shape[-1], frame.shape[-2]])
    return np.array(frame[:, :, p1[1]:p2[1], p1[0]:p2[0]])

def cut_rois(frame, rois):
    return [cut_roi(frame, roi) for roi in rois]

def resize_input(frame, target_shape):
    assert len(frame.shape) == len(target_shape), \
        "Expected a frame with %s dimensions, but got %s" % \
        (len(target_shape), len(frame.shape))

    assert frame.shape[0] == 1, "Only batch size 1 is supported"
    n, c, h, w = target_shape

    input = frame[0]
    if not np.array_equal(target_shape[-2:], frame.shape[-2:]):
        input = input.transpose((1, 2, 0)) # to HWC
        input = cv2.resize(input, (w, h))
        input = input.transpose((2, 0, 1)) # to CHW

    return input.reshape((n, c, h, w))

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

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

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep    

def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def align_face(face_frame, landmarks):
    
    # ref: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks

    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]     # right eye, left eye Y
    dx = right_eye[0] - left_eye[0]  # right eye, left eye X
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    ##eyes_center = ((right_eye[0] + left_eye[0]) // 2, (right_eye[1] + left_eye[1]) // 2)

    # center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))

    return aligned_face

def plot_boxes(boxes, confs, clses, image):
    for index, box in enumerate(boxes):
        x1, y1, x2, y2= box
        #cast to int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = clses[index]
        conf = confs[index]
        #generate three random values using numpy
        #color = np.random.randint(0, 256, size=3).tolist()

        text = f'{conf:.2f}'
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
        image = cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return image
