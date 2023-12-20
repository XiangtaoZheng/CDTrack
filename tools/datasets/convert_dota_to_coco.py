import math

import cv2
import json
import os
import xml.etree.ElementTree as et
import numpy as np
import pickle

# categories = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane', 'ship', 'soccer-ball-field',
#  'basketball-court', 'ground-track-field', 'small-vehicle', 'baseball-diamond', 'tennis-court',
#  'roundabout', 'storage-tank', 'harbor')

# def convert_dota_to_coco():
#     dataset_dir = '/home/user1/dataset/DOTA_SPLIT/'
#     split = 'val'
#
#     gt_path = os.path.join(dataset_dir, split, 'annfiles/patch_annfile.pkl')
#     groundtruth = open(gt_path, 'rb')
#     groundtruth = pickle.load(groundtruth)
#     categories = groundtruth['cls']
#     ship_id = categories.index('ship')
#     plane_id = categories.index('plane')
#     groundtruth = groundtruth['content']
#
#     output_path = os.path.join(dataset_dir, 'annotations', 'ship_{}.json'.format(split))
#     output = {'images': [], 'annotations': [],
#               'categories': [{'id': 1, 'name': 'ship'}], 'videos': [{'id': 1, 'file_name': 'DOTA'}]}
#
#     image_id = 1
#     ann_id = 1
#     conf, iscrowd = 1.0, 0
#     for i in range(len(groundtruth)):
#         gt = groundtruth[i]
#         filename = gt['filename']
#         bbox = gt['ann']['bboxes']
#         width, height = gt['width'], gt['height']
#         labels = gt['ann']['labels']
#         if ship_id not in labels:
#             continue
#         image_path = os.path.join(dataset_dir, split, 'images', filename)
#         image_info = {
#             'file_name': image_path,
#             'id': image_id,
#             'frame_id': 1,
#             'prev_image_id': -1,
#             'next_image_id': -1,
#             'video_id': 1,
#             'height': height,
#             'width': width,
#         }
#         output['images'].append(image_info)
#         track_id = 1
#         for j in range(len(bbox)):
#             if labels[j] != ship_id:
#                 continue
#             box = bbox[j]
#             l, t, r, b = int(np.min(box[0::2])), int(np.min(box[1::2])), \
#                          int(np.max(box[0::2])), int(np.max(box[1::2]))
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
#         image_id += 1
#     print('total {} images, {} samples'.format(image_id - 1, ann_id - 1))
#     json.dump(output, open(output_path, 'w'))


def convert_dota_to_coco():
    dataset_dir = '/home/user6/dataset/DOTA/'
    split = 'val'

    gt_path = os.path.join(dataset_dir, split, 'annfiles/patch_annfile.pkl')
    groundtruth = open(gt_path, 'rb')
    groundtruth = pickle.load(groundtruth)
    categories = groundtruth['cls']
    # ship_id = categories.index('ship')
    plane_id = categories.index('plane')
    groundtruth = groundtruth['content']

    output_path = os.path.join(dataset_dir, 'annotations', 'plane_{}.json'.format(split))
    output = {'images': [], 'annotations': [],
              'categories': [{'id': 1, 'name': 'plane'}], 'videos': [{'id': 1, 'file_name': 'DOTA'}]}

    image_id = 1
    ann_id = 1
    conf, iscrowd = 1.0, 0
    total_absolate_area = 0
    for i in range(len(groundtruth)):
        gt = groundtruth[i]
        filename = gt['filename']
        bbox = gt['ann']['bboxes']
        width, height = gt['width'], gt['height']
        labels = gt['ann']['labels']
        if plane_id not in labels:
            continue
        image_path = os.path.join(dataset_dir, split, 'images', filename)
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
        track_id = 1
        for j in range(len(bbox)):
            if labels[j] != plane_id:
                continue
            box = bbox[j]
            l, t, r, b = int(np.min(box[0::2])), int(np.min(box[1::2])), \
                         int(np.max(box[0::2])), int(np.max(box[1::2]))
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
        image_id += 1
    print('total {} images, {} samples'.format(image_id - 1, ann_id - 1))
    print('total absolate area: {}, average absolate area: {}'.format(total_absolate_area,
                                                                      total_absolate_area / (ann_id - 1)))
    json.dump(output, open(output_path, 'w'))


def check_xml():
    gt_path = '/home/user1/dataset/DOTA_SPLIT/val/annfiles/patch_annfile.pkl'
    groundtruth = open(gt_path, 'rb')
    groundtruth = pickle.load(groundtruth)
    categories = groundtruth['cls']
    ship_id = categories.index('ship')
    plane_id = categories.index('plane')
    # 'gsd', 'filename', 'ann', 'x_start', 'y_start', 'id', 'ori_id', 'height', 'width'
    groundtruth = groundtruth['content']
    image_id = 0
    while True:
        image_id += 1
        gt = groundtruth[image_id]
        filename = gt['filename']
        bbox = gt['ann']['bboxes']
        width, height = gt['width'], gt['height']
        labels = gt['ann']['labels']
        if plane_id not in labels:
            continue
        file_path = os.path.join('/home/user1/dataset/DOTA_SPLIT/val/images', filename)
        img = cv2.imread(file_path)
        for i in range(len(bbox)):
            box = bbox[i]
            l, t, r, b = int(np.min(box[0::2])), int(np.min(box[1::2])), \
                         int(np.max(box[0::2])), int(np.max(box[1::2]))
            category = int(labels[i])
            print(categories[category])
            cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
        img = cv2.resize(img, (1024, 768))
        cv2.imshow('img', img)
        cv2.waitKey(0)


def check_json():
    json_path = '/home/user1/dataset/DOTA_SPLIT/annotations/plane_dota.json'
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

def mix_dota():
    dota_train = '/home/user1/dataset/DOTA_SPLIT/annotations/plane_train.json'
    dota_val = '/home/user1/dataset/DOTA_SPLIT/annotations/plane_val.json'
    dota_train, dota_val = json.load(open(dota_train, 'r')), json.load(open(dota_val, 'r'))

    output_path = '/home/user1/dataset/DOTA_SPLIT/annotations/plane_dota.json'
    output = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'plane'}],
              'videos': [{'id': 1, 'file_name': 'DOTA'}]}

    output['images'] = dota_train['images']
    output['annotations'] = dota_train['annotations']
    total_images, total_anns = len(output['images']), len(output['annotations'])

    for image in dota_val['images']:
        image['id'] += total_images
        image['video_id'] = 1
        output['images'].append(image)

    for ann in dota_val['annotations']:
        ann['id'] += total_anns
        ann['image_id'] += total_images
        output['annotations'].append(ann)

    total_images += len(dota_val['images'])
    total_anns += len(dota_val['annotations'])

    print('total {} images, {} objects'.format(total_images, total_anns))
    json.dump(output, open(output_path, 'w'))

# ship: total 2473 images, 76153 objects
# plane: total 2593 images, 18788 objects
if __name__ == '__main__':
    convert_dota_to_coco()
    # check_xml()
    # check_json()
    # mix_dota()
