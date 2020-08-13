import cv2
from config import *
from math import exp as exp
from logo.image_iterator import ImageIterator


class YoloParams:
    def __init__(self, param, side):
        self.num = 9
        self.coords = 4
        self.classes = 1
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, h_scale, w_scale):
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    width = int(w * w_scale)
    height = int(h * h_scale)
    return [xmin, ymin, width, height]


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    boxes = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                boxes.append(scale_bbox(x=x, y=y, h=h, w=w, h_scale=orig_im_h, w_scale=orig_im_w))
    return boxes


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def detect(exec_net, net, images, conf_threshold, is_async_mode=False):
    logos, boxes = list(), list()
    cur_request_id = 0
    next_request_id = 1
    images = ImageIterator(images=images)

    while images.isOpened():
        if is_async_mode:
            ret, next_image = images.read()
        else:
            ret, image = images.read()

        if not ret:
            break

        if is_async_mode:
            request_id = next_request_id
            custom_image = cv2.resize(next_image, (480, 480), interpolation=cv2.INTER_LINEAR)
        else:
            request_id = cur_request_id
            custom_image = cv2.resize(image, (480, 480), interpolation=cv2.INTER_LINEAR)
        custom_image = custom_image.transpose(2, 0, 1)
        custom_image = custom_image.reshape(1, 3, 480, 480)
        exec_net.start_async(request_id=request_id, inputs={'inputs': custom_image})

        if exec_net.requests[cur_request_id].wait(-1) == 0:
            output = exec_net.requests[cur_request_id].outputs

            for layer_name, out_blob in output.items():
                out_blob = out_blob.reshape(net.layers[net.layers[layer_name].parents[0]].out_data[0].shape)
                layer_params = YoloParams(net.layers[layer_name].params, out_blob.shape[2])
                if is_async_mode:
                    boxes += parse_yolo_region(out_blob, custom_image.shape[2:], next_image.shape[:-1], layer_params,
                                               conf_threshold)
                else:
                    boxes += parse_yolo_region(out_blob, custom_image.shape[2:], image.shape[:-1], layer_params,
                                               conf_threshold)

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            image = next_image

        indices = cv2.dnn.NMSBoxes(boxes, [1]*len(boxes), conf_threshold, nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            logos.append(image[y:y+h, x:x+w])

    return logos
