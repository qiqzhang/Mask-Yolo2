import numpy as np
import tensorflow as tf
import os
import cv2


def get_images_labels_list():
    images_list = []
    labels_list = []
    with open("/home/linx-dl/hzr/2012_train.txt", 'r') as f:
        for line in f:
            image_full_name = line.split('\n')[0]
            # print(image_full_name)
            images_list.append(image_full_name)
            label_name = line.split('/')[-1].split('.')[0] + '.txt'
            label_full_name = os.path.join("/home/linx-dl/hzr/VOCdevkit/VOC2012/labels", label_name)

            #  print(label_full_name)
            labels_list.append(label_full_name)
    return images_list, labels_list



images_list,labels_list  = get_images_labels_list()


def parse_function(filename, label):
    image_string = tf.read_file(filename)

    label = tf.read_file(label)
    # label = tf.decode_raw(label,tf.uint8)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [416, 416])
    return image, label

dataset = tf.data.Dataset.from_tensor_slices((tf.constant(images_list),tf.constant(labels_list)))
#dataset = dataset.map(lambda filename, label: tuple(tf.py_func(_read_py_function, [filename, label], [tf.uint8, tf.string])),num_parallel_calls=4)
dataset = dataset.map(parse_function,num_parallel_calls=4)
dataset = dataset.shuffle(len(images_list))
dataset = dataset.batch(10)
dataset = dataset.prefetch(1)

iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)

    batch_images, batch_labels = sess.run(iterator.get_next())



image_list=[]
for i,label in enumerate(batch_labels):
    label_list=[]
    for j,v in enumerate(label.decode().splitlines()):
        label_list.append([float(vv)for vv in v.split(' ')])
    image_list.append(label_list)


def iou_wh(r1, r2):
    min_w = min(r1[0], r2[0])
    min_h = min(r1[1], r2[1])
    area_r1 = r1[0] * r1[1]
    area_r2 = r2[0] * r2[1]

    intersect = min_w * min_h
    union = area_r1 + area_r2 - intersect

    return intersect / union


def get_active_anchors(roi, anchors):
    indxs = []
    iou_max, index_max = 0, 0

    for i, a in enumerate(anchors):
        iou = iou_wh(roi[1:], a)
        if iou > 0.7:
            indxs.append(i)
        if iou > iou_max:
            iou_max, index_max = iou, i

    if len(indxs) == 0:
        indxs.append(index_max)

    return indxs


def get_grid_cell(roi):
    x_center = roi[1] + roi[3] / 2.0
    y_center = roi[2] + roi[4] / 2.0

    grid_x = int(x_center / float(1 / 13))
    grid_y = int(y_center / float(1 / 13))

    return grid_x, grid_y


def roi2label(roi, anchor):
    x_center = roi[1] + roi[3] / 2.0
    y_center = roi[2] + roi[4] / 2.0

    grid_x = x_center / float(1 / 13)
    grid_y = y_center / float(1 / 13)

    grid_x_offset = grid_x - int(grid_x)
    grid_y_offset = grid_y - int(grid_y)

    roi_w_scale = roi[3] / (1 / anchor[0])
    roi_h_scale = roi[4] / (1 / anchor[1])

    label = [grid_x_offset, grid_y_offset, roi_w_scale, roi_h_scale]

    return label


label = np.zeros([10, 13, 13, 5, 6], dtype=np.float32)
anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
anchor_list = []
for i in range(0, len(anchors), 2):
    anchor_list.append([anchors[i] / 13, anchors[i + 1] / 13])
print(anchor_list)
for image_index in range(len(image_list)):  # every image
    for label_index in range(len(image_list[image_index])):  # every label
        # print(image_list[image_index][label_index])
        activate_anchor_index = get_active_anchors(image_list[image_index][label_index], anchor_list)

        grid_x, grid_y = get_grid_cell(image_list[image_index][label_index])
        for index in activate_anchor_index:
            anchor_label = roi2label(image_list[image_index][label_index], anchor_list[index])
            label[image_index, grid_x, grid_y, index] = np.concatenate(
                (anchor_label, [int(image_list[image_index][label_index][0])], [1.0]))

print(label[0,...])
