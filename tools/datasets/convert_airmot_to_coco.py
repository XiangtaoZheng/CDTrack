import math

import numpy as np
import os
import json
import cv2
import random
from PIL import Image


# def convert_airmot_to_coco():
#     dataset_dir = '/home/user1/dataset/AIRMOT/'
#     split = 'train'
#     gt_dir = os.path.join(dataset_dir, 'gt')
#
#     video_dir = os.path.join(dataset_dir, '{}'.format(split))
#
#     output = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'ship'}],
#               'videos': []}
#     output_path = os.path.join(dataset_dir, 'annotations', 'ship_{}.json'.format(split))
#
#     video_id = 1
#     image_id = 1
#     ann_id = 1
#     iscrowd, conf = 0, 1.0
#     diff_id = 0
#     total_absolate_area = 0
#     ship_video = []
#
#     for seq in sorted(os.listdir(video_dir)):
#         # video = seq.split('.')[0]
#         # if video not in select_video:
#         #     continue
#         image_dir = os.path.join(video_dir, seq, 'img')
#
#         gt_path = os.path.join(gt_dir, '{}.txt'.format(seq))
#         gt = np.loadtxt(gt_path, delimiter=',')
#         if len(gt) == 0:
#             continue
#         if len(gt[np.where(gt[:, 7] == 2)]) < 1:
#             continue
#         ship_video.append(seq)
#         ship_gt = gt[np.where(gt[:, 7] == 2)]
#         diff_id += len(np.unique(ship_gt[:, 1]))
#
#         output['videos'].append({'id': video_id, 'file_name': seq})
#         video_length = len(os.listdir(image_dir))
#
#         for frame in range(1, video_length + 1):
#             image_path = os.path.join(image_dir, '{:06d}.jpg'.format(frame))
#             prev_image_id = image_id - 1 if frame > 1 else -1
#             next_image_id = image_id + 1 if frame < video_length else -1
#             width, height = Image.open(image_path).size
#             image_info = {
#                 'file_name': image_path,
#                 'id': image_id,
#                 'frame_id': frame,
#                 'prev_image_id': prev_image_id,
#                 'next_image_id': next_image_id,
#                 'video_id': video_id,
#                 'height': height,
#                 'width': width,
#             }
#             output['images'].append(image_info)
#
#             anns = gt[np.where(gt[:, 0] == frame)]
#             for ann in anns:
#                 category = int(ann[7])
#                 if category != 2:
#                     continue
#                 track_id = int(ann[1])
#                 bbox = ann[2:6]
#                 area = bbox[2] * bbox[3]
#                 ann_info = {
#                     'id': ann_id,
#                     'category_id': 1,
#                     'image_id': image_id,
#                     'track_id': track_id,
#                     'bbox': bbox.tolist(),
#                     'conf': conf,
#                     'iscrowd': iscrowd,
#                     'area': area
#                 }
#                 output['annotations'].append(ann_info)
#                 ann_id += 1
#                 total_absolate_area += math.sqrt(area)
#             image_id += 1
#         video_id += 1
#     print('total {} images, {} samples, {} diff samples'.format(image_id - 1, ann_id - 1, diff_id))
#     print('total absolate area: {}, average absolate area: {}'.format(total_absolate_area,
#                                                                       total_absolate_area / (ann_id - 1)))
#     print(ship_video)
#     # json.dump(output, open(output_path, 'w'))


def convert_airmot_to_coco():
    dataset_dir = '/home/user1/dataset/AIRMOT/'
    split = 'test'
    gt_dir = os.path.join(dataset_dir, 'gt')

    video_dir = os.path.join(dataset_dir, '{}_plane'.format(split))

    output = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'plane'}],
              'videos': []}
    output_path = os.path.join(dataset_dir, 'annotations', 'plane_{}.json'.format(split))

    video_id = 1
    image_id = 1
    ann_id = 1
    iscrowd, conf = 0, 1.0
    diff_id = 0
    total_absolate_area = 0
    plane_video = []
    if split == 'train':
        select_video = ['11', '3', '35', '36', '4', '44', '46', '47', '49', '52', '54', '55', '56', '66', '72', '74']
    else:
        select_video = ['1', '2', '30', '31', '39', '41', '43', '48', '50', '53', '64', '69', '7', '75', '8']

    for seq in sorted(os.listdir(video_dir)):
        # video = seq.split('.')[0]
        if seq not in select_video:
            continue
        image_dir = os.path.join(video_dir, seq, 'img')

        gt_path = os.path.join(gt_dir, '{}.txt'.format(seq))
        gt = np.loadtxt(gt_path, delimiter=',')
        if len(gt[np.where(gt[:, 7] == 1)]) < 1:
            continue
        plane_video.append(seq)
        plane_gt = gt[np.where(gt[:, 7] == 1)]
        diff_id += len(np.unique(plane_gt[:, 1]))

        output['videos'].append({'id': video_id, 'file_name': seq})
        video_length = len(os.listdir(image_dir))

        for frame in range(1, video_length + 1):
            image_path = os.path.join(image_dir, '{:06d}.jpg'.format(frame))
            prev_image_id = image_id - 1 if frame > 1 else -1
            next_image_id = image_id + 1 if frame < video_length else -1
            width, height = Image.open(image_path).size
            image_info = {
                'file_name': image_path,
                'id': image_id,
                'frame_id': frame,
                'prev_image_id': prev_image_id,
                'next_image_id': next_image_id,
                'video_id': video_id,
                'height': height,
                'width': width,
            }
            output['images'].append(image_info)

            anns = gt[np.where(gt[:, 0] == frame)]
            for ann in anns:
                category = int(ann[7])
                if category != 1:
                    continue
                track_id = int(ann[1])
                bbox = ann[2:6]
                area = bbox[2] * bbox[3]
                ann_info = {
                    'id': ann_id,
                    'category_id': 1,
                    'image_id': image_id,
                    'track_id': track_id,
                    'bbox': bbox.tolist(),
                    'conf': conf,
                    'iscrowd': iscrowd,
                    'area': area
                }
                output['annotations'].append(ann_info)
                ann_id += 1
                total_absolate_area += math.sqrt(area)
            image_id += 1
        video_id += 1
    print('total {} images, {} samples, {} diff samples'.format(image_id - 1, ann_id - 1, diff_id))
    print('total absolate area: {}, average absolate area: {}'.format(total_absolate_area,
                                                                      total_absolate_area / (ann_id - 1)))
    print(plane_video)
    json.dump(output, open(output_path, 'w'))


def random_select():
    # ship_video = ['12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '33', '38', '40', '50', '51', '52', '61', '63', '65', '70', '77', '79', '80']
    # results = random.sample(ship_video, 3)
    # print(results)

    plane_video = ['1', '11', '2', '3', '30', '31', '35', '36', '39', '4', '41', '43', '44', '46', '47', '48', '49', '50', '52', '53',
     '54', '55', '56', '64', '66', '69', '7', '72', '74', '75', '8']
    # results = random.sample(plane_video, 16)
    # print(sorted(results))



def check_json():
    json_path = '/home/user1/dataset/AIRMOT/annotations/plane_train.json'
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


def single_category_video():
    category = 'plane'
    dataset_dir = '/home/user1/dataset/AIRMOT/'
    gt_dir = os.path.join(dataset_dir, 'gt')
    videos = []
    for seq in sorted(os.listdir(gt_dir)):
        video = seq.split('.')[0]
        gt_path = os.path.join(gt_dir, seq)
        gt = np.loadtxt(gt_path, delimiter=',')
        if len(gt[np.where(gt[:, 7] == 1)]) < 1:
            continue
        videos.append(video)
    print(len(videos), videos)

if __name__ == '__main__':
    # convert_airmot_to_coco()
    # random_select()
    check_json()
    # single_category_video()