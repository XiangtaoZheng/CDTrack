import json
import os
import numpy as np
import cv2
import random

def mix_dior_dota():
    dior_trainval = '/home/user2/dataset/DIOR/annotations/ship_trainval.json'
    dior_test = '/home/user2/dataset/DIOR/annotations/ship_test.json'
    dota_train = '/home/user2/dataset/DOTA/annotations/ship_train.json'
    dota_val = '/home/user2/dataset/DOTA/annotations/ship_val.json'
    dior_trainval, dior_test, dota_train, dota_val = json.load(open(dior_trainval, 'r')), json.load(open(dior_test, 'r')),\
                                                     json.load(open(dota_train, 'r')), json.load(open(dota_val, 'r'))

    output_path = '/home/user2/dataset/AIRMOT/annotations/dior_dota.json'
    output = {'images': [], 'annotations': [], 'categories': [{'id': 1, 'name': 'plane'}, {'id': 2, 'name': 'ship'}],
              'videos': [{'id': 1, 'file_name': 'DIOR'}, {'id': 2, 'file_name': 'DOTA'}]}

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

    for image in dota_train['images']:
        image['id'] += total_images
        image['video_id'] = 2
        output['images'].append(image)

    for ann in dota_train['annotations']:
        ann['id'] += total_anns
        ann['image_id'] += total_images
        output['annotations'].append(ann)

    total_images += len(dota_train['images'])
    total_anns += len(dota_train['annotations'])

    for image in dota_val['images']:
        image['id'] += total_images
        image['video_id'] = 2
        output['images'].append(image)

    for ann in dota_val['annotations']:
        ann['id'] += total_anns
        ann['image_id'] += total_images
        output['annotations'].append(ann)

    total_images += len(dota_val['images'])
    total_anns += len(dota_val['annotations'])

    print('total {} images, {} objects'.format(total_images, total_anns))
    json.dump(output, open(output_path, 'w'))

def check_json():
    json_path = '/home/user2/dataset/AIRMOT/annotations/dota_plane.json'
    gt = json.load(open(json_path, 'r'))

    for i in range(len(gt['images'])):
        ann = random.choice(gt['images'])
        image_path = ann['file_name']
        image_id = ann['id']
        img = cv2.imread(image_path)

        for j in range(len(gt['annotations'])):
            if gt['annotations'][j]['image_id'] == image_id:
                bbox = gt['annotations'][j]['bbox']
                l, t, r, b = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(img, (l, t), (l + r, t + b), (0, 0, 255), 1)

        img = cv2.resize(img, (1024, 768))
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    # mix_dior_dota()
    check_json()
