#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import cv2
import numpy as np

from yolox.utils import adjust_box_anns

import random

from ..data_augment import box_candidates, random_perspective, augment_hsv, WeakTransform, StrongTransform
from .datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class AdaptiveMosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
            self, dataset, img_size, mosaic=True, preproc=None,
            degrees=10.0, translate=0.1, scale=(0.5, 1.5), mscale=(0.5, 1.5),
            shear=2.0, perspective=0.0, enable_mixup=True, source=True, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            scale (tuple):
            mscale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mscale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup

        self.source = source
        self.weakTransform = WeakTransform(rgb_means=(0.485, 0.456, 0.406),
                                           std=(0.229, 0.224, 0.225),
                                           max_labels=1000)
        self.strongTransform = StrongTransform(rgb_means=(0.485, 0.456, 0.406),
                                               std=(0.229, 0.224, 0.225),
                                               max_labels=1000)

    def __len__(self):
        return len(self._dataset)

    @Dataset.resize_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic:
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            img, _labels, _, _ = self._dataset.pull_item(idx)
            h0, w0 = img.shape[:2]  # orig hw
            scale = min(1. * input_h / h0, 1. * input_w / w0)
            mosaic_img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )

            labels = _labels.copy()
            # Normalized xywh to pixel xyxy format
            if _labels.size > 0:
                labels[:, 0] = scale * _labels[:, 0]
                labels[:, 1] = scale * _labels[:, 1]
                labels[:, 2] = scale * _labels[:, 2]
                labels[:, 3] = scale * _labels[:, 3]
            mosaic_labels = labels

            if len(mosaic_labels):
                mosaic_labels = mosaic_labels[mosaic_labels[:, 0] < input_w]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 2] > 0]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 1] < input_h]
                mosaic_labels = mosaic_labels[mosaic_labels[:, 3] > 0]

            # augment_hsv(mosaic_img)
            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[0, 0],
            )  # border to remove

            if self.source:
                mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
                # mix_img, padded_labels = mosaic_img, mosaic_labels
                img_info = (mix_img.shape[1], mix_img.shape[0])
                return mix_img, padded_labels, img_info, np.array([idx])
            else:
                flip = random.randrange(2)
                weak_img, _ = self.weakTransform(mosaic_img, mosaic_labels, self.input_dim, flip)
                strong_img, _ = self.strongTransform(mosaic_img, mosaic_labels, self.input_dim, flip)
                img_info = (weak_img.shape[1], weak_img.shape[0])
                return weak_img, strong_img, img_info, np.array([idx])

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, id_ = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, id_

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3)) * 114.0
        else:
            cp_img = np.ones(input_dim) * 114.0
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[
        : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor
        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)
        ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                    origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        '''
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        '''
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            id_labels = cp_labels[keep_list, 5:6].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels, id_labels))
            # remove outside bbox
            labels = labels[labels[:, 0] < target_w]
            labels = labels[labels[:, 2] > 0]
            labels = labels[labels[:, 1] < target_h]
            labels = labels[labels[:, 3] > 0]
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img, origin_labels
