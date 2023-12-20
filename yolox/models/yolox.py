#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn
import torch

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None, source=True, task=''):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs



class DAYOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test
    """

    def __init__(self, backbone=None, head=None, da=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.da = da

    # def forward(self, x, targets=None):
    #     if isinstance(x, list):
    #         source_feature = self.backbone(x[0])
    #         target_feature = self.backbone(x[1])
    #     else:
    #         source_feature = self.backbone(x)
    #
    #     if self.training:
    #         assert targets is not None
    #         da_loss_img_s, da_loss_img_t = self.da(source_feature, target_feature)
    #
    #         loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
    #             source_feature, targets, x[0]
    #         )
    #         da_loss = da_loss_img_s + da_loss_img_t
    #         loss += da_loss
    #
    #         outputs = {
    #             "total_loss": loss,
    #             "iou_loss": iou_loss,
    #             "l1_loss": l1_loss,
    #             "conf_loss": conf_loss,
    #             "cls_loss": cls_loss,
    #             "da_loss": da_loss,
    #             "source_loss": da_loss_img_s,
    #             "target_loss": da_loss_img_t,
    #             "num_fg": num_fg,
    #         }
    #     else:
    #         outputs = self.head(source_feature)
    #     return outputs

    def forward(self, x, targets=None):
        if isinstance(x, list):
            source_feature = self.backbone(x[0])
            target_feature = self.backbone(x[1])
        else:
            source_feature = self.backbone(x)

        if self.training:
            assert targets is not None
            da_loss_img_s = self.da(source_feature, source=True)
            da_loss_img_t = self.da(target_feature, source=False)

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                source_feature, targets, x[0]
            )
            da_loss = da_loss_img_s + da_loss_img_t
            loss += da_loss

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "da_loss": da_loss,
                "source_loss": da_loss_img_s,
                "target_loss": da_loss_img_t,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(source_feature)
        return outputs


# class AdaptiveYOLOX(nn.Module):
#     """
#     YOLOX model module. The module list is defined by create_yolov3_modules function.
#     The network returns loss values from three YOLO layers during training
#     and detection results during test
#     """
#
#     def __init__(self, backbone=None, head=None, da=None):
#         super().__init__()
#         self.backbone = backbone
#         self.head = head
#         self.da = da
#
#     def forward(self, x, targets=None, task='supervise', source=True):
#         if not self.training:
#             feature = self.backbone(x)
#             outputs = self.head(feature)
#             return outputs
#
#         if task == 'supervise':
#             feature = self.backbone(x)
#             if source:
#                 da_loss_s = self.da(feature, source=source)
#             else:
#                 da_loss_s = 0
#             loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
#                 feature, targets, x
#             )
#             da_loss_s *= 0.001
#             loss += da_loss_s
#             outputs = {
#                 "total_loss": loss,
#                 "iou_loss": iou_loss,
#                 "l1_loss": l1_loss,
#                 "conf_loss": conf_loss,
#                 "cls_loss": cls_loss,
#                 "da_loss_s": da_loss_s
#             }
#             return outputs
#
#         elif task == 'domain':
#             source_feature = self.backbone(x[0])
#             target_feature = self.backbone(x[1])
#             da_loss_img_s = self.da(source_feature, source=True)
#             da_loss_img_t = self.da(target_feature, source=False)
#             da_loss = da_loss_img_s + da_loss_img_t
#             outputs = {
#                 "da_loss": da_loss,
#                 "da_loss_s": da_loss_img_s,
#                 "da_loss_t": da_loss_img_t,
#             }
#             return outputs

class AdaptiveYOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test
    """

    def __init__(self, backbone=None, head=None, da=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.da = da

    def forward(self, x, targets=None, task='supervise', source=True):
        if not self.training:
            feature = self.backbone(x)
            outputs = self.head(feature)

            if self.head.vis_cam:
                return feature, outputs
            return outputs

        if task == 'supervise':
            feature = self.backbone(x)
            da_loss = self.da(feature, source=source)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                feature, targets, x
            )
            loss += (da_loss * 0.1)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                # "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "da_loss": da_loss
            }
            return outputs

        elif task == 'domain':
            source_feature = self.backbone(x[0])
            target_feature = self.backbone(x[1])
            da_loss_img_s = self.da(source_feature, source=True)
            da_loss_img_t = self.da(target_feature, source=False)
            da_loss = da_loss_img_s + da_loss_img_t
            outputs = {
                "da_loss": da_loss,
                "da_loss_s": da_loss_img_s,
                "da_loss_t": da_loss_img_t,
            }
            return outputs