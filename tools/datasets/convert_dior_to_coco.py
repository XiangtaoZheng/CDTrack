import math

import cv2
import json
import os
import xml.etree.ElementTree as et
import numpy as np


# def convert_dior_to_coco():
#     dataset_dir = '/home/user1/dataset/DIOR/'
#     split = 'test'
#
#     output_path = os.path.join(dataset_dir, 'annotations', 'ship_{}.json'.format(split))
#     output = {'images': [], 'annotations': [],
#            'categories': [{'id': 1, 'name': 'ship'}], 'videos': [{'id': 1, 'file_name': 'DIOR'}]}
#
#     gt_dir = os.path.join(dataset_dir, 'gt')
#
#     if split != 'trainval':
#         split_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '{}.txt'.format(split))
#         split_file = open(split_file, 'r').readlines()
#         split_file = [i.split('\n')[0] for i in split_file]
#     else:
#         train_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '{}.txt'.format('train'))
#         val_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '{}.txt'.format('val'))
#         train_file = open(train_file, 'r').readlines()
#         val_file = open(val_file, 'r').readlines()
#         train_file = [i.split('\n')[0] for i in train_file]
#         val_file = [i.split('\n')[0] for i in val_file]
#         split_file = train_file + val_file
#     print(len(split_file))
#
#     image_id = 1
#     ann_id = 1
#     conf, iscrowd = 1.0, 0
#     for seq in sorted(split_file):
#         image_path = os.path.join(dataset_dir, split if split == 'test' else 'trainval', '{}.jpg'.format(seq))
#         gt_path = os.path.join(gt_dir, '{}.xml'.format(seq))
#         tree = et.parse(gt_path)
#         root = tree.getroot()
#         exist_object = False
#         track_id = 1
#         for obj in root.findall('object'):
#             category = obj.find('name').text
#             if category != 'ship':
#                 continue
#             exist_object = True
#             bbox = list(obj.iter('bndbox'))[0]
#             l, t, r, b = int(bbox.find('xmin').text), int(bbox.find('ymin').text), \
#                          int(bbox.find('xmax').text), int(bbox.find('ymax').text)
#             w, h = r - l, b - t
#             ann_info = {
#                 'id': ann_id,
#                 'category_id': 1,
#                 'image_id': image_id,
#                 'track_id': track_id,
#                 'bbox': [l, t, w, h],
#                 'conf': conf,
#                 'iscrowd': iscrowd,
#                 'area': w * h
#             }
#             output['annotations'].append(ann_info)
#             track_id += 1
#             ann_id += 1
#         if exist_object:
#             size = list(root.iter('size'))[0]
#             width, height = int(size.find('width').text), int(size.find('height').text)
#             image_info = {
#                 'file_name': image_path,
#                 'id': image_id,
#                 'frame_id': 1,
#                 'prev_image_id': -1,
#                 'next_image_id': -1,
#                 'video_id': 1,
#                 'height': height,
#                 'width': width,
#             }
#             output['images'].append(image_info)
#             image_id += 1
#     print('total {} images, {} samples'.format(image_id - 1, ann_id - 1))
#     json.dump(output, open(output_path, 'w'))


def convert_dior_to_coco():
    dataset_dir = '/home/user6/dataset/DIOR/'
    split = 'test'

    output_path = os.path.join(dataset_dir, 'annotations', 'plane_{}.json'.format(split))
    output = {'images': [], 'annotations': [],
              'categories': [{'id': 1, 'name': 'plane'}], 'videos': [{'id': 1, 'file_name': 'DIOR'}]}

    gt_dir = os.path.join(dataset_dir, 'gt')

    if split != 'trainval':
        split_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '{}.txt'.format(split))
        split_file = open(split_file, 'r').readlines()
        split_file = [i.split('\n')[0] for i in split_file]
    else:
        train_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '{}.txt'.format('train'))
        val_file = os.path.join(dataset_dir, 'ImageSets', 'Main', '{}.txt'.format('val'))
        train_file = open(train_file, 'r').readlines()
        val_file = open(val_file, 'r').readlines()
        train_file = [i.split('\n')[0] for i in train_file]
        val_file = [i.split('\n')[0] for i in val_file]
        split_file = train_file + val_file
    print(len(split_file))

    image_id = 1
    ann_id = 1
    conf, iscrowd = 1.0, 0
    total_absolate_area = 0
    for seq in sorted(split_file):
        image_path = os.path.join(dataset_dir, split if split == 'test' else 'trainval', '{}.jpg'.format(seq))
        gt_path = os.path.join(gt_dir, '{}.xml'.format(seq))
        tree = et.parse(gt_path)
        root = tree.getroot()
        exist_object = False
        track_id = 1
        for obj in root.findall('object'):
            category = obj.find('name').text
            if category != 'airplane':
                continue
            exist_object = True
            bbox = list(obj.iter('bndbox'))[0]
            l, t, r, b = int(bbox.find('xmin').text), int(bbox.find('ymin').text), \
                         int(bbox.find('xmax').text), int(bbox.find('ymax').text)
            w, h = r - l, b - t
            ann_info = {
                'id': ann_id,
                'category_id': 1,
                'image_id': image_id,
                'track_id': track_id,
                'bbox': [l, t, w, h],
                'conf': conf,
                'iscrowd': iscrowd,
                'area': w * h
            }
            output['annotations'].append(ann_info)
            track_id += 1
            ann_id += 1
            total_absolate_area += math.sqrt(w * h)
        if exist_object:
            size = list(root.iter('size'))[0]
            width, height = int(size.find('width').text), int(size.find('height').text)
            image_info = {
                'file_name': image_path,
                'id': image_id,
                'frame_id': 1,
                'prev_image_id': -1,
                'next_image_id': -1,
                'video_id': 1,
                'height': height,
                'width': width,
            }
            output['images'].append(image_info)
            image_id += 1
    print('total {} images, {} samples'.format(image_id - 1, ann_id - 1))
    print('total absolate area: {}, average absolate area: {}'.format(total_absolate_area,
                                                                      total_absolate_area / (ann_id - 1)))
    json.dump(output, open(output_path, 'w'))


def check_xml():
    # DIOR
    image_id = 16
    xml_path = '/home/user2/dataset/DIOR/gt/{:05d}.xml'.format(image_id)
    img_path = '/home/user2/dataset/DIOR/trainval/{:05d}.jpg'.format(image_id)
    img = cv2.imread(img_path)
    tree = et.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    size = list(root.iter('size'))[0]
    width, height = size.find('width').text, size.find('height').text
    for obj in root.findall('object'):
        category = obj.find('name').text
        if category != 'ship':
            continue
        bbox = list(obj.iter('bndbox'))[0]
        l, t, r, b = int(bbox.find('xmin').text), int(bbox.find('ymin').text), \
                     int(bbox.find('xmax').text), int(bbox.find('ymax').text)
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def check_json():
    # json_path = '/home/user1/dataset/DIOR/annotations/plane_dior.json'
    json_path = '/home/user6/dataset/AIRMOT/annotations/dior_plane.json'
    gt = json.load(open(json_path, 'r'))

    for i in range(len(gt['images'])):
        image_path = gt['images'][i]['file_name']
        image_id = gt['images'][i]['id']
        img = cv2.imread(image_path)

        for j in range(len(gt['annotations'])):
            if gt['annotations'][j]['image_id'] == image_id:
                bbox = gt['annotations'][j]['bbox']
                l, t, r, b = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(img, (l, t), (l + r, t + b), (0, 0, 255), 1)

        img = cv2.resize(img, (1024, 768))
        cv2.imshow('img', img)
        cv2.waitKey(0)


def mix_dior():
    dior_trainval = '/home/user1/dataset/DIOR/annotations/plane_trainval.json'
    dior_test = '/home/user1/dataset/DIOR/annotations/plane_test.json'
    dior_trainval, dior_test = json.load(open(dior_trainval, 'r')), json.load(open(dior_test, 'r'))

    output_path = '/home/user1/dataset/DIOR/annotations/plane_dior.json'
    output = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'plane'}],
              'videos': [{'id': 1, 'file_name': 'DIOR'}]}

    output['images'] = dior_trainval['images']
    output['annotations'] = dior_trainval['annotations']
    total_images, total_anns = len(output['images']), len(output['annotations'])

    for image in dior_test['images']:
        image['id'] += total_images
        image['video_id'] = 1
        output['images'].append(image)

    for ann in dior_test['annotations']:
        ann['id'] += total_anns
        ann['image_id'] += total_images
        output['annotations'].append(ann)

    total_images += len(dior_test['images'])
    total_anns += len(dior_test['annotations'])

    print('total {} images, {} objects'.format(total_images, total_anns))
    json.dump(output, open(output_path, 'w'))


# ship: total 2702 images, 62400 objects
# plane: total 1387 images, 10104 objects
if __name__ == '__main__':
    convert_dior_to_coco()
    # check_xml()
    # check_json()
    # mix_dior()
