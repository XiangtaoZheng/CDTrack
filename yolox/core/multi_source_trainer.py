#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch
import cv2
import numpy as np
import copy

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    postprocess
)

import datetime
import os
import time


# class MultiTrainer:
#     def __init__(self, exp, args):
#         # init function only defines some basic attr, other attrs like model, optimizer are built in
#         # before_train methods.
#         self.exp = exp
#         self.args = args
#
#         # training related attr
#         self.max_epoch = exp.max_epoch
#         self.amp_training = args.fp16
#         self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
#         self.is_distributed = get_world_size() > 1
#         self.rank = get_rank()
#         self.local_rank = args.local_rank
#         self.device = "cuda:{}".format(self.local_rank)
#         self.use_model_ema = exp.ema
#
#         # data/dataloader related attr
#         self.data_type = torch.float16 if args.fp16 else torch.float32
#         self.input_size = exp.input_size
#         self.best_ap_1, self.best_ap_2 = 0, 0
#         self.iter_loss = {}
#
#         # metric record
#         self.meter = MeterBuffer(window_size=exp.print_interval)
#         self.file_name = os.path.join(exp.output_dir, args.experiment_name)
#
#         if self.rank == 0:
#             os.makedirs(self.file_name, exist_ok=True)
#
#         setup_logger(
#             self.file_name,
#             distributed_rank=self.rank,
#             filename="train_log.txt",
#             mode="a",
#         )
#
#     def train(self):
#         self.before_train()
#         # '''
#         try:
#             self.train_in_epoch()
#         except Exception:
#             raise
#         finally:
#             self.after_train()
#         # '''
#
#     def train_in_epoch(self):
#         for self.epoch in range(self.start_epoch, self.max_epoch):
#             self.before_epoch()
#             self.train_in_iter()
#             self.after_epoch()
#
#     def train_in_iter(self):
#         for self.iter in range(self.max_iter):
#             self.before_iter()
#             self.train_one_iter()
#             self.after_iter()
#
#     def train_one_iter(self):
#         iter_start_time = time.time()
#
#         source_inps_1, source_targets_1 = self.source_prefetcher_1.next()
#         source_targets_1 = source_targets_1[:, :, :5]
#         source_inps_1 = source_inps_1.to(self.data_type)
#         source_targets_1 = source_targets_1.to(self.data_type)
#         source_targets_1.requires_grad = False
#
#         source_inps_2, source_targets_2 = self.source_prefetcher_2.next()
#         source_targets_2 = source_targets_2[:, :, :5]
#         source_inps_2 = source_inps_2.to(self.data_type)
#         source_targets_2 = source_targets_2.to(self.data_type)
#         source_targets_2.requires_grad = False
#
#         target_weak, target_strong = self.target_prefetcher.next()
#         target_weak = target_weak.to(self.data_type)
#         target_strong = target_strong.to(self.data_type)
#
#         data_end_time = time.time()
#
#         with torch.cuda.amp.autocast(enabled=self.amp_training):
#             if self.epoch < self.exp.supervise_epoch:
#                 outputs = {}
#                 model_1_outputs = self.model_1(source_inps_1, source_targets_1, task='supervise')
#                 model_2_outputs = self.model_2(source_inps_2, source_targets_2, task='supervise')
#                 outputs.update(model_1_outputs)
#                 for k, v in model_2_outputs.items():
#                     outputs[k + '_2'] = model_2_outputs[k]
#                     if k == 'total_loss':
#                         outputs['total_loss'] += model_2_outputs['total_loss']
#             else:
#                 if self.epoch == self.exp.supervise_epoch and self.iter == 0:
#                     logger.info('Start Adaptive Training!')
#                     self.update_teacher_model(keep_rate=0, k=1)
#                     self.update_teacher_model(keep_rate=0, k=2)
#                 else:
#                     if self.iter % self.exp.update_teacher_iter == 0:
#                         self.update_teacher_model(keep_rate=self.exp.ema_keep_rate, k=1)
#                         self.update_teacher_model(keep_rate=self.exp.ema_keep_rate, k=2)
#
#                 outputs = {}
#
#                 # ## supervise label data
#                 model_1_label_outputs = self.model_1(source_inps_1, source_targets_1, task='supervise')
#                 model_2_label_outputs = self.model_2(source_inps_2, source_targets_2, task='supervise')
#                 outputs['total_loss'] = model_1_label_outputs['total_loss'] + model_2_label_outputs['total_loss']
#                 for k, v in model_1_label_outputs.items():
#                     outputs[k + '_label_1'] = model_1_label_outputs[k]
#                     outputs[k + '_label_2'] = model_2_label_outputs[k]
#
#                 batch_size = target_weak.shape[0]
#                 pseudo_label_1 = torch.zeros((batch_size, 1000, 5)).to(self.data_type).cuda()
#                 pseudo_label_1.requires_grad = False
#                 pseudo_label_2 = torch.zeros((batch_size, 1000, 5)).to(self.data_type).cuda()
#                 pseudo_label_2.requires_grad = False
#                 pseudo_score_1 = torch.zeros((batch_size, 1000, 2)).to(self.data_type).cuda()
#                 pseudo_score_2 = torch.zeros((batch_size, 1000, 2)).to(self.data_type).cuda()
#                 num_pseudo_labels_1, num_pseudo_labels_2 = 0, 0
#                 with torch.no_grad():
#                     self.modelTeacher_1.eval()
#                     self.modelTeacher_2.eval()
#                     for i in range(batch_size):
#                         predictions_1 = self.modelTeacher_1(target_weak[i].unsqueeze(0))
#                         predictions_1 = postprocess(predictions_1, num_classes=1,
#                                                     conf_thre=self.exp.pseudo_label_thresh,
#                                                     nms_thre=self.exp.nmsthre)
#                         predictions_1 = predictions_1[0]
#                         if predictions_1 is not None:
#                             pseudo_label_1[i, :len(predictions_1), 0] = predictions_1[:, 6]
#                             pseudo_label_1[i, :len(predictions_1), 3:5] = predictions_1[:, 2:4] - \
#                                                                           predictions_1[:, 0:2]
#                             pseudo_label_1[i, :len(predictions_1), 1:3] = predictions_1[:, 0:2] + \
#                                                                           pseudo_label_1[i, :len(predictions_1),
#                                                                           3:5] / 2
#                             pseudo_score_1[i, :len(predictions_1), :] = predictions_1[:, 4:6]
#                             num_pseudo_labels_1 += len(predictions_1)
#
#                         predictions_2 = self.modelTeacher_2(target_weak[i].unsqueeze(0))
#                         predictions_2 = postprocess(predictions_2, num_classes=1,
#                                                     conf_thre=self.exp.pseudo_label_thresh,
#                                                     nms_thre=self.exp.nmsthre)
#                         predictions_2 = predictions_2[0]
#                         if predictions_2 is not None:
#                             pseudo_label_2[i, :len(predictions_2), 0] = predictions_2[:, 6]
#                             pseudo_label_2[i, :len(predictions_2), 3:5] = predictions_2[:, 2:4] - \
#                                                                           predictions_2[:, 0:2]
#                             pseudo_label_2[i, :len(predictions_2), 1:3] = predictions_2[:, 0:2] + \
#                                                                           pseudo_label_2[i, :len(predictions_2),
#                                                                           3:5] / 2
#                             pseudo_score_2[i, :len(predictions_2), :] = predictions_2[:, 4:6]
#                             num_pseudo_labels_2 += len(predictions_2)
#
#                 # ## visualize pseudo label
#                 # for i in range(batch_size):
#                 #     img_strong = target_strong[i].cpu().numpy().transpose(1, 2, 0) * 255.0
#                 #     img_strong = img_strong.astype(np.uint8).copy()
#                 #     img_weak = target_weak[i].cpu().numpy().transpose(1, 2, 0) * 255.0
#                 #     img_weak = img_weak.astype(np.uint8).copy()
#                 #     bboxes = pseudo_label_1[i].cpu().numpy()
#                 #     scores = pseudo_score_1[i].cpu().numpy()
#                 #     for j in range(len(bboxes)):
#                 #         bbox, score = bboxes[j], scores[j]
#                 #         cx, cy, w, h = bbox[1], bbox[2], bbox[3], bbox[4]
#                 #         l, t, r, b = round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)
#                 #         conf_score, cls_score = round(float(score[0]), 2), round(float(score[1]), 2)
#                 #         score_text = '{} {}'.format(conf_score, cls_score)
#                 #         category_text = '{}'.format(bbox[0])
#                 #         # cv2.rectangle(img_strong, (l, t), (r, b), (0, 0, 255), 1)
#                 #         cv2.rectangle(img_weak, (l, t), (r, b), (0, 0, 255), 1)
#                 #         cv2.putText(img_weak, score_text, (l, t), cv2.FONT_HERSHEY_PLAIN,
#                 #                     1, (0, 0, 255), thickness=1)
#                 #     bboxes = pseudo_label_2[i].cpu().numpy()
#                 #     scores = pseudo_score_2[i].cpu().numpy()
#                 #     for j in range(len(bboxes)):
#                 #         bbox, score = bboxes[j], scores[j]
#                 #         cx, cy, w, h = bbox[1], bbox[2], bbox[3], bbox[4]
#                 #         l, t, r, b = round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)
#                 #         conf_score, cls_score = round(float(score[0]), 2), round(float(score[1]), 2)
#                 #         score_text = '{} {}'.format(conf_score, cls_score)
#                 #         category_text = '{}'.format(bbox[0])
#                 #         # cv2.rectangle(img_strong, (l, t), (r, b), (0, 0, 255), 1)
#                 #         cv2.rectangle(img_strong, (l, t), (r, b), (0, 0, 255), 1)
#                 #         cv2.putText(img_strong, score_text, (l, t), cv2.FONT_HERSHEY_PLAIN,
#                 #                     1, (0, 0, 255), thickness=1)
#                 #     cv2.imshow('img_strong', cv2.resize(img_strong, (1280, 736)))
#                 #     cv2.imshow('img_weak', cv2.resize(img_weak, (1280, 736)))
#                 #     cv2.waitKey(0)
#
#                 if num_pseudo_labels_2 > 0:
#                     model_1_unlabel_outputs = self.model_1(target_strong, pseudo_label_2, task='supervise',
#                                                            source=False)
#                     outputs['total_loss'] += model_1_unlabel_outputs['total_loss'] * self.exp.unlabel_loss_weight
#                     for k, v in model_1_unlabel_outputs.items():
#                         outputs[k + '_unlabel_1'] = model_1_unlabel_outputs[k] * self.exp.unlabel_loss_weight
#
#                 if num_pseudo_labels_1 > 0:
#                     model_2_unlabel_outputs = self.model_2(target_strong, pseudo_label_1, task='supervise',
#                                                            source=False)
#                     outputs['total_loss'] += model_2_unlabel_outputs['total_loss'] * self.exp.unlabel_loss_weight
#                     for k, v in model_2_unlabel_outputs.items():
#                         outputs[k + '_unlabel_2'] = model_2_unlabel_outputs[k] * self.exp.unlabel_loss_weight
#
#                 # model_1_da_outputs = self.model_1([source_inps_1, target_strong], task='domain')
#                 # model_2_da_outputs = self.model_2([source_inps_2, target_strong], task='domain')
#                 # outputs['total_loss'] += model_1_da_outputs['da_loss'] * self.exp.dis_loss_weight + model_2_da_outputs[
#                 #     'da_loss'] * self.exp.dis_loss_weight
#                 # for k, v in model_1_da_outputs.items():
#                 #     outputs[k + '_1'] = model_1_da_outputs[k] * self.exp.dis_loss_weight
#                 #     outputs[k + '_2'] = model_2_da_outputs[k] * self.exp.dis_loss_weight
#
#         loss = outputs["total_loss"]
#
#         self.optimizer.zero_grad()
#         self.scaler.scale(loss).backward(retain_graph=True)
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
#
#         if self.use_model_ema:
#             self.ema_model_1.update(self.model_1)
#             self.ema_model_2.update(self.model_2)
#
#         lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
#         for param_group in self.optimizer.param_groups:
#             param_group["lr"] = lr
#
#         iter_end_time = time.time()
#         self.meter.update(
#             iter_time=iter_end_time - iter_start_time,
#             data_time=data_end_time - iter_start_time,
#             lr=lr,
#             **outputs,
#         )
#
#     def update_teacher_model(self, keep_rate=0, k=1):
#         if k == 1:
#             student_model_dict = self.model_1.state_dict()
#             new_teacher_dict = OrderedDict()
#             for key, value in self.modelTeacher_1.state_dict().items():
#                 if key in student_model_dict.keys():
#                     new_teacher_dict[key] = (
#                             student_model_dict[key] *
#                             (1 - keep_rate) + value * keep_rate
#                     )
#                 else:
#                     raise Exception("{} is not found in student model".format(key))
#             self.modelTeacher_1.load_state_dict(new_teacher_dict)
#         else:
#             student_model_dict = self.model_2.state_dict()
#             new_teacher_dict = OrderedDict()
#             for key, value in self.modelTeacher_2.state_dict().items():
#                 if key in student_model_dict.keys():
#                     new_teacher_dict[key] = (
#                             student_model_dict[key] *
#                             (1 - keep_rate) + value * keep_rate
#                     )
#                 else:
#                     raise Exception("{} is not found in student model".format(key))
#             self.modelTeacher_2.load_state_dict(new_teacher_dict)
#
#     def before_train(self):
#         logger.info("args: {}".format(self.args))
#         logger.info("exp value:\n{}".format(self.exp))
#
#         # model related init
#         torch.cuda.set_device(self.local_rank)
#         model_1, modelTeacher_1 = self.exp.get_model()
#         model_2, modelTeacher_2 = copy.deepcopy(model_1), copy.deepcopy(modelTeacher_1)
#
#         model_1.to(self.device)
#         modelTeacher_1.to(self.device)
#         model_2.to(self.device)
#         modelTeacher_2.to(self.device)
#
#         # solver related init
#         self.optimizer = self.exp.get_optimizer(self.args.batch_size)
#
#         # value of epoch will be set in `resume_train`
#         ckpt_1, ckpt_2 = self.args.ckpt.split('-')
#         model_1 = self.resume_train(model_1, ckpt_1)
#         model_2 = self.resume_train(model_2, ckpt_2)
#
#         # data related init
#         self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
#
#         self.source_train_loader_1 = self.exp.get_data_loader(
#             batch_size=self.args.batch_size,
#             is_distributed=self.is_distributed,
#             ann=self.exp.source_ann_1,
#             no_aug=self.no_aug, source=True
#         )
#
#         self.source_train_loader_2 = self.exp.get_data_loader(
#             batch_size=self.args.batch_size,
#             is_distributed=self.is_distributed,
#             ann=self.exp.source_ann_2,
#             no_aug=self.no_aug, source=True
#         )
#
#         self.target_train_loader = self.exp.get_data_loader(
#             batch_size=self.args.batch_size,
#             is_distributed=self.is_distributed,
#             ann=self.exp.target_ann,
#             no_aug=self.no_aug, source=False
#         )
#
#         # state_dict_1 = model_1.state_dict()
#         # state_dict_2 = model_2.state_dict()
#         # for k, v in state_dict_1.items():
#         #     print('state_dict_1: ', state_dict_1[k])
#         #     break
#         # for k, v in state_dict_2.items():
#         #     print('state_dict_2: ', state_dict_2[k])
#         #     break
#
#         # for i, batch in enumerate(self.target_train_loader):
#         #     for j in batch:
#         #         print(j.shape if isinstance(j, torch.Tensor) else j)
#         #     img = batch[0].squeeze(0).numpy()
#         #     img = img.astype(np.uint8)
#         #     img = img.copy()
#         #     for bbox in batch[1][0]:
#         #         l = int(bbox[0])
#         #         t = int(bbox[1])
#         #         r = int(bbox[2])
#         #         b = int(bbox[3])
#         #         cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
#         #     img = cv2.resize(img, (1280, 768))
#         #     cv2.imshow('img', img)
#         #     cv2.waitKey(0)
#
#         # for i, batch in enumerate(self.target_train_loader):
#         #     for j in batch:
#         #         print(j.shape if isinstance(j, torch.Tensor) else j)
#         #     img_weak = batch[0].squeeze(0).numpy().astype(np.uint8)
#         #     img_weak = img_weak.copy()
#         #     img_strong = batch[0].squeeze(0).numpy().astype(np.uint8)
#         #     img_strong = img_strong.copy()
#         #     img_weak = cv2.resize(img_weak, (1280, 736))
#         #     img_strong = cv2.resize(img_strong, (1280, 736))
#         #     cv2.imshow('img_weak', img_weak)
#         #     cv2.imshow('img_strong', img_strong)
#         #     cv2.waitKey(0)
#
#         # '''
#
#         logger.info("init prefetcher, this might take one minute or less...")
#         self.target_prefetcher = DataPrefetcher(self.target_train_loader)
#         self.source_prefetcher_1 = DataPrefetcher(self.source_train_loader_1)
#         self.source_prefetcher_2 = DataPrefetcher(self.source_train_loader_2)
#         # max_iter means iters per epoch
#         # self.max_iter = max(len(self.source_train_loader), len(self.target_train_loader))
#         self.max_iter = max(len(self.source_train_loader_1), len(self.source_train_loader_2))
#         # self.max_iter = len(self.target_train_loader)
#
#         self.lr_scheduler = self.exp.get_lr_scheduler(
#             self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
#         )
#         if self.args.occupy:
#             occupy_mem(self.local_rank)
#
#         if self.is_distributed:
#             model_1 = DDP(model_1, device_ids=[self.local_rank], broadcast_buffers=False)
#             modelTeacher_1 = DDP(modelTeacher_1, device_ids=[self.local_rank], broadcast_buffers=False)
#             model_2 = DDP(model_2, device_ids=[self.local_rank], broadcast_buffers=False)
#             modelTeacher_2 = DDP(modelTeacher_2, device_ids=[self.local_rank], broadcast_buffers=False)
#
#         if self.use_model_ema:
#             self.ema_model_1 = ModelEMA(model_1, 0.9998)
#             self.ema_model_1.updates = self.max_iter * self.start_epoch
#             self.ema_model_2 = ModelEMA(model_2, 0.9998)
#             self.ema_model_2.updates = self.max_iter * self.start_epoch
#
#         self.model_1, self.model_2 = model_1, model_2
#         self.modelTeacher_1, self.modelTeacher_2 = modelTeacher_1, modelTeacher_2
#         self.model_1.train()
#         self.modelTeacher_1.train()
#         self.model_2.train()
#         self.modelTeacher_2.train()
#
#         self.evaluator = self.exp.get_evaluator(
#             batch_size=self.args.batch_size, is_distributed=self.is_distributed, testdev=True
#         )
#         # Tensorboard logger
#         if self.rank == 0:
#             self.tblogger = SummaryWriter(self.file_name)
#
#         logger.info("Training start...")
#         # logger.info("\n{}".format(model))
#         # '''
#
#     def after_train(self):
#         logger.info(
#             "Training of experiment is done and the best AP is {:.2f}".format(
#                 max(self.best_ap_1 * 100, self.best_ap_2 * 100)
#             )
#         )
#
#     def before_epoch(self):
#         logger.info("---> start train epoch{}".format(self.epoch + 1))
#
#         if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
#             logger.info("--->No mosaic aug now!")
#             self.source_train_loader_1.close_mosaic()
#             self.source_train_loader_2.close_mosaic()
#             self.target_train_loader.close_mosaic()
#             logger.info("--->Add additional L1 loss now!")
#             if self.is_distributed:
#                 self.model_1.module.head.use_l1 = True
#                 self.model_2.module.head.use_l1 = True
#             else:
#                 self.model_1.head.use_l1 = True
#                 self.model_2.head.use_l1 = True
#
#             self.exp.eval_interval = 1
#             if not self.no_aug:
#                 self.save_ckpt(ckpt_name="last_mosaic_epoch", k=1)
#                 self.save_ckpt(ckpt_name="last_mosaic_epoch", k=2)
#
#     def after_epoch(self):
#         if self.use_model_ema:
#             self.ema_model_1.update_attr(self.model_1)
#             self.ema_model_2.update_attr(self.model_2)
#
#         self.save_ckpt(ckpt_name="latest", k=1)
#         self.save_ckpt(ckpt_name="latest", k=2)
#
#         if (self.epoch + 1) == self.exp.supervise_epoch:
#             self.save_ckpt(ckpt_name="last_supervise", k=1)
#             self.save_ckpt(ckpt_name="last_supervise", k=2)
#
#         if (self.epoch + 1) % 10 == 0:
#             self.save_ckpt(ckpt_name="{}".format(self.epoch + 1), k=1)
#             self.save_ckpt(ckpt_name="{}".format(self.epoch + 1), k=2)
#
#         if (self.epoch + 1) < self.exp.supervise_epoch:
#             if (self.epoch + 1) % self.exp.eval_interval == 0:
#                 all_reduce_norm(self.model_1)
#                 all_reduce_norm(self.model_2)
#                 self.evaluate_and_save_model()
#         else:
#             if (self.epoch + 1) % 1 == 0:
#                 all_reduce_norm(self.model_1)
#                 all_reduce_norm(self.model_2)
#                 self.evaluate_and_save_model()
#
#     def before_iter(self):
#         pass
#
#     def after_iter(self):
#         """
#         `after_iter` contains two parts of logic:
#             * log information
#             * reset setting of resize
#         """
#         # log needed information
#
#         if (self.iter + 1) % self.exp.print_interval == 0:
#             # TODO check ETA logic
#             left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
#             eta_seconds = self.meter["iter_time"].global_avg * left_iters
#             eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
#
#             progress_str = "epoch: {}/{}, iter: {}/{}".format(
#                 self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
#             )
#             loss_meter = self.meter.get_filtered_meter("loss")
#
#             loss_str = ", ".join(
#                 ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
#             )
#             for k, v in loss_meter.items():
#                 if k not in self.iter_loss:
#                     self.iter_loss[k] = [v.latest]
#                 else:
#                     self.iter_loss[k].append(v.latest)
#
#             # loss_str = ", "
#             # for k, v in loss_meter.items():
#             #     if v.latest is not None:
#             #         loss_str.join("{}: {:.3f}".format(k, v.latest))
#             #     else:
#             #         loss_str.join("{}: None".format(k))
#             #
#             # for k, v in loss_meter.items():
#             #     if k not in self.iter_loss:
#             #         if v.latest is not None:
#             #             self.iter_loss[k] = [v.latest]
#             #         else:
#             #             self.iter_loss[k] = [0]
#             #     else:
#             #         if v.latest is not None:
#             #             self.iter_loss[k].append(v.latest)
#             #         else:
#             #             self.iter_loss[k].append(0)
#
#             time_meter = self.meter.get_filtered_meter("time")
#             time_str = ", ".join(
#                 ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
#             )
#
#             logger.info(
#                 "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
#                     progress_str,
#                     gpu_mem_usage(),
#                     time_str,
#                     loss_str,
#                     self.meter["lr"].latest,
#                 )
#                 + (", size: {:d}, {}".format(self.input_size[0], eta_str))
#             )
#             self.meter.clear_meters()
#
#         # random resizing
#         if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
#             self.input_size = self.exp.random_resize(
#                 self.source_train_loader_1, self.source_train_loader_2, self.target_train_loader, self.epoch, self.rank,
#                 self.is_distributed
#             )
#
#     @property
#     def progress_in_iter(self):
#         return self.epoch * self.max_iter + self.iter
#
#     def resume_train(self, model, ckpt_file):
#         if self.args.resume:
#             logger.info("resume training")
#             # if self.args.ckpt is None:
#             #     ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
#             # else:
#             #     ckpt_file = self.args.ckpt
#
#             ckpt = torch.load(ckpt_file, map_location=self.device)
#             # resume the model/optimizer state dict
#             model.load_state_dict(ckpt["model"])
#             self.optimizer.load_state_dict(ckpt["optimizer"])
#             start_epoch = (
#                 self.args.start_epoch - 1
#                 if self.args.start_epoch is not None
#                 else ckpt["start_epoch"]
#             )
#             self.start_epoch = start_epoch
#             logger.info(
#                 "loaded checkpoint '{}' (epoch {})".format(
#                     self.args.resume, self.start_epoch
#                 )
#             )  # noqa
#         else:
#             if self.args.ckpt is not None:
#                 # ckpt_file = self.args.ckpt
#                 ckpt = torch.load(ckpt_file, map_location=self.device)
#                 logger.info("loading {} checkpoint for fine tuning".format(ckpt['start_epoch']))
#                 ckpt = ckpt["model"]
#                 model = load_ckpt(model, ckpt)
#             self.start_epoch = 0
#         return model
#
#     def evaluate_and_save_model(self):
#         evalmodel = self.ema_model_1.ema if self.use_model_ema else self.model_1
#         ap50_95, ap50, summary = self.exp.eval(
#             evalmodel, self.evaluator, self.is_distributed
#         )
#         self.model_1.train()
#         if self.rank == 0:
#             self.tblogger.add_scalar("val_1/COCOAP50", ap50, self.epoch + 1)
#             self.tblogger.add_scalar("val_1/COCOAP50_95", ap50_95, self.epoch + 1)
#             for k, v in self.iter_loss.items():
#                 loss = sum(self.iter_loss[k]) / len(self.iter_loss[k])
#                 self.tblogger.add_scalar("train/{}".format(k), loss, self.epoch + 1)
#             logger.info("\n" + summary)
#         synchronize()
#
#         # self.best_ap = max(self.best_ap, ap50_95)
#         self.save_ckpt("last_epoch", ap50 > self.best_ap_1, k=1)
#         self.best_ap_1 = max(self.best_ap_1, ap50)
#
#         evalmodel = self.ema_model_2.ema if self.use_model_ema else self.model_2
#         ap50_95, ap50, summary = self.exp.eval(
#             evalmodel, self.evaluator, self.is_distributed
#         )
#         self.model_2.train()
#         if self.rank == 0:
#             self.tblogger.add_scalar("val_2/COCOAP50", ap50, self.epoch + 1)
#             self.tblogger.add_scalar("val_2/COCOAP50_95", ap50_95, self.epoch + 1)
#             logger.info("\n" + summary)
#         synchronize()
#
#         # self.best_ap = max(self.best_ap, ap50_95)
#         self.save_ckpt("last_epoch", ap50 > self.best_ap_2, k=2)
#         self.best_ap_2 = max(self.best_ap_2, ap50)
#
#     def save_ckpt(self, ckpt_name, update_best_ckpt=False, k=1):
#         if self.rank == 0:
#             if k == 1:
#                 save_model = self.ema_model_1.ema if self.use_model_ema else self.model_1
#             else:
#                 save_model = self.ema_model_2.ema if self.use_model_ema else self.model_2
#             logger.info("Save weights to {}".format(self.file_name))
#             ckpt_state = {
#                 "start_epoch": self.epoch + 1,
#                 "model": save_model.state_dict(),
#                 "optimizer": self.optimizer.state_dict(),
#             }
#             save_checkpoint(
#                 ckpt_state,
#                 update_best_ckpt,
#                 self.file_name,
#                 ckpt_name + '_{}'.format(k) if not update_best_ckpt else '{}'.format(k),
#             )


class MultiTrainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap_1, self.best_ap_2 = 0, 0
        self.iter_loss = {}

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        # '''
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()
        # '''

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        source_inps_1, source_targets_1 = self.source_prefetcher_1.next()
        source_targets_1 = source_targets_1[:, :, :5]
        source_inps_1 = source_inps_1.to(self.data_type)
        source_targets_1 = source_targets_1.to(self.data_type)
        source_targets_1.requires_grad = False

        source_inps_2, source_targets_2 = self.source_prefetcher_2.next()
        source_targets_2 = source_targets_2[:, :, :5]
        source_inps_2 = source_inps_2.to(self.data_type)
        source_targets_2 = source_targets_2.to(self.data_type)
        source_targets_2.requires_grad = False

        target_weak, target_strong = self.target_prefetcher.next()
        target_weak = target_weak.to(self.data_type)
        target_strong = target_strong.to(self.data_type)

        data_end_time = time.time()

        # with torch.cuda.amp.autocast(enabled=self.amp_training):
        outputs = {}
        if self.epoch < self.exp.supervise_epoch:
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                model_1_outputs = self.model_1(source_inps_1, source_targets_1, task='supervise')
                model_2_outputs = self.model_2(source_inps_2, source_targets_2, task='supervise')
            outputs.update(model_1_outputs)
            for k, v in model_2_outputs.items():
                outputs[k + '_2'] = model_2_outputs[k]
                if k == 'total_loss':
                    outputs['total_loss'] += model_2_outputs['total_loss']
        else:
            if self.epoch == self.exp.supervise_epoch and self.iter == 0:
                logger.info('Start Adaptive Training!')
                self.update_teacher_model(keep_rate=0, k=1)
                self.update_teacher_model(keep_rate=0, k=2)
            else:
                if self.iter % self.exp.update_teacher_iter == 0:
                    self.update_teacher_model(keep_rate=self.exp.ema_keep_rate, k=1)
                    self.update_teacher_model(keep_rate=self.exp.ema_keep_rate, k=2)

            ## generate pseudo labels
            batch_size = target_weak.shape[0]
            pseudo_label_1 = torch.zeros((batch_size, 1000, 5)).to(self.data_type).cuda()
            pseudo_label_1.requires_grad = False
            pseudo_label_2 = torch.zeros((batch_size, 1000, 5)).to(self.data_type).cuda()
            pseudo_label_2.requires_grad = False
            pseudo_score_1 = torch.zeros((batch_size, 1000, 2)).to(self.data_type).cuda()
            pseudo_score_2 = torch.zeros((batch_size, 1000, 2)).to(self.data_type).cuda()
            num_pseudo_labels_1, num_pseudo_labels_2 = 0, 0
            with torch.cuda.amp.autocast(enabled=self.amp_training):
                with torch.no_grad():
                    self.modelTeacher_1.eval()
                    self.modelTeacher_2.eval()
                    for i in range(batch_size):
                        predictions_1 = self.modelTeacher_1(target_weak[i].unsqueeze(0))
                        predictions_1 = postprocess(predictions_1, num_classes=1,
                                                    conf_thre=self.exp.pseudo_label_thresh,
                                                    nms_thre=self.exp.nmsthre)
                        predictions_1 = predictions_1[0]
                        if predictions_1 is not None:
                            pseudo_label_1[i, :len(predictions_1), 0] = predictions_1[:, 6]
                            pseudo_label_1[i, :len(predictions_1), 3:5] = predictions_1[:, 2:4] - \
                                                                          predictions_1[:, 0:2]
                            pseudo_label_1[i, :len(predictions_1), 1:3] = predictions_1[:, 0:2] + \
                                                                          pseudo_label_1[i, :len(predictions_1),
                                                                          3:5] / 2
                            pseudo_score_1[i, :len(predictions_1), :] = predictions_1[:, 4:6]
                            num_pseudo_labels_1 += len(predictions_1)

                        predictions_2 = self.modelTeacher_2(target_weak[i].unsqueeze(0))
                        predictions_2 = postprocess(predictions_2, num_classes=1,
                                                    conf_thre=self.exp.pseudo_label_thresh,
                                                    nms_thre=self.exp.nmsthre)
                        predictions_2 = predictions_2[0]
                        if predictions_2 is not None:
                            pseudo_label_2[i, :len(predictions_2), 0] = predictions_2[:, 6]
                            pseudo_label_2[i, :len(predictions_2), 3:5] = predictions_2[:, 2:4] - \
                                                                          predictions_2[:, 0:2]
                            pseudo_label_2[i, :len(predictions_2), 1:3] = predictions_2[:, 0:2] + \
                                                                          pseudo_label_2[i, :len(predictions_2),
                                                                          3:5] / 2
                            pseudo_score_2[i, :len(predictions_2), :] = predictions_2[:, 4:6]
                            num_pseudo_labels_2 += len(predictions_2)

            '''
            # ## visualize pseudo label
            if self.iter == 0:
                for i in range(1):
                    img_strong = target_strong[i].cpu().numpy().transpose(1, 2, 0) * 255.0
                    img_strong = img_strong.astype(np.uint8).copy()
                    img_weak = target_weak[i].cpu().numpy().transpose(1, 2, 0) * 255.0
                    img_weak = img_weak.astype(np.uint8).copy()
                    bboxes = pseudo_label_1[i].cpu().numpy()
                    scores = pseudo_score_1[i].cpu().numpy()
                    for j in range(len(bboxes)):
                        bbox, score = bboxes[j], scores[j]
                        cx, cy, w, h = bbox[1], bbox[2], bbox[3], bbox[4]
                        l, t, r, b = round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)
                        conf_score, cls_score = round(float(score[0]), 2), round(float(score[1]), 2)
                        score_text = '{} {}'.format(conf_score, cls_score)
                        category_text = '{}'.format(bbox[0])
                        # cv2.rectangle(img_strong, (l, t), (r, b), (0, 0, 255), 1)
                        cv2.rectangle(img_weak, (l, t), (r, b), (0, 0, 255), 1)
                        cv2.putText(img_weak, score_text, (l, t), cv2.FONT_HERSHEY_PLAIN,
                                    1, (0, 0, 255), thickness=1)
                    bboxes = pseudo_label_2[i].cpu().numpy()
                    scores = pseudo_score_2[i].cpu().numpy()
                    for j in range(len(bboxes)):
                        bbox, score = bboxes[j], scores[j]
                        cx, cy, w, h = bbox[1], bbox[2], bbox[3], bbox[4]
                        l, t, r, b = round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2)
                        conf_score, cls_score = round(float(score[0]), 2), round(float(score[1]), 2)
                        score_text = '{} {}'.format(conf_score, cls_score)
                        category_text = '{}'.format(bbox[0])
                        # cv2.rectangle(img_strong, (l, t), (r, b), (0, 0, 255), 1)
                        cv2.rectangle(img_strong, (l, t), (r, b), (0, 0, 255), 1)
                        cv2.putText(img_strong, score_text, (l, t), cv2.FONT_HERSHEY_PLAIN,
                                    1, (0, 0, 255), thickness=1)
                    # cv2.imshow('img_strong', cv2.resize(img_strong, (1280, 736)))
                    # cv2.imshow('img_weak', cv2.resize(img_weak, (1280, 736)))
                    # cv2.waitKey(0)
                    save_dir = '/home/user6/code/TGRS/BoT-SORT/visualization/'
                    cv2.imwrite(save_dir + 'epoch_{}_img_strong.jpg'.format(self.epoch), img_strong)
                    cv2.imwrite(save_dir + 'epoch_{}_img_weak.jpg'.format(self.epoch), img_weak)
            '''

            with torch.cuda.amp.autocast(enabled=self.amp_training):
                # ## supervise source with labels
                model_1_label_outputs = self.model_1(source_inps_1, source_targets_1, task='supervise')
                model_2_label_outputs = self.model_2(source_inps_2, source_targets_2, task='supervise')
                outputs['total_loss'] = model_1_label_outputs['total_loss'] + model_2_label_outputs['total_loss']
                for k, v in model_1_label_outputs.items():
                    outputs[k + '_label_1'] = model_1_label_outputs[k]
                    outputs[k + '_label_2'] = model_2_label_outputs[k]

                # ## supervise target with pseudo labels
                if num_pseudo_labels_2 > 0:
                    model_1_unlabel_outputs = self.model_1(target_strong, pseudo_label_2, task='supervise',
                                                           source=False)
                    outputs['total_loss'] += model_1_unlabel_outputs['total_loss'] * self.exp.unlabel_loss_weight
                    for k, v in model_1_unlabel_outputs.items():
                        outputs[k + '_unlabel_1'] = model_1_unlabel_outputs[k] * self.exp.unlabel_loss_weight

                if num_pseudo_labels_1 > 0:
                    model_2_unlabel_outputs = self.model_2(target_strong, pseudo_label_1, task='supervise',
                                                           source=False)
                    outputs['total_loss'] += model_2_unlabel_outputs['total_loss'] * self.exp.unlabel_loss_weight
                    for k, v in model_2_unlabel_outputs.items():
                        outputs[k + '_unlabel_2'] = model_2_unlabel_outputs[k] * self.exp.unlabel_loss_weight

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model_1.update(self.model_1)
            self.ema_model_2.update(self.model_2)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def update_teacher_model(self, keep_rate=0, k=1):
        if k == 1:
            student_model_dict = self.model_1.state_dict()
            new_teacher_dict = OrderedDict()
            for key, value in self.modelTeacher_1.state_dict().items():
                if key in student_model_dict.keys():
                    new_teacher_dict[key] = (
                            student_model_dict[key] *
                            (1 - keep_rate) + value * keep_rate
                    )
            self.modelTeacher_1.load_state_dict(new_teacher_dict)
        else:
            student_model_dict = self.model_2.state_dict()
            new_teacher_dict = OrderedDict()
            for key, value in self.modelTeacher_2.state_dict().items():
                if key in student_model_dict.keys():
                    new_teacher_dict[key] = (
                            student_model_dict[key] *
                            (1 - keep_rate) + value * keep_rate
                    )
            self.modelTeacher_2.load_state_dict(new_teacher_dict)

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model_1, modelTeacher_1 = self.exp.get_model()
        model_2, modelTeacher_2 = copy.deepcopy(model_1), copy.deepcopy(modelTeacher_1)

        model_1.to(self.device)
        modelTeacher_1.to(self.device)
        model_2.to(self.device)
        modelTeacher_2.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        ckpt_1, ckpt_2 = self.args.ckpt.split('-')
        model_1 = self.resume_train(model_1, ckpt_1)
        model_2 = self.resume_train(model_2, ckpt_2)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs

        self.source_train_loader_1 = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            ann=self.exp.source_ann_1,
            no_aug=self.no_aug, source=True
        )

        self.source_train_loader_2 = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            ann=self.exp.source_ann_2,
            no_aug=self.no_aug, source=True
        )

        self.target_train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            ann=self.exp.target_ann,
            no_aug=self.no_aug, source=False
        )

        # state_dict_1 = model_1.state_dict()
        # state_dict_2 = model_2.state_dict()
        # for k, v in state_dict_1.items():
        #     print('state_dict_1: ', state_dict_1[k])
        #     break
        # for k, v in state_dict_2.items():
        #     print('state_dict_2: ', state_dict_2[k])
        #     break

        # for i, batch in enumerate(self.source_train_loader_2):
        #     for j in batch:
        #         print(j.shape if isinstance(j, torch.Tensor) else j)
        #     img = batch[0].squeeze(0).numpy()
        #     img = img.astype(np.uint8)
        #     img = img.copy()
        #     for bbox in batch[1][0]:
        #         l = int(bbox[0])
        #         t = int(bbox[1])
        #         r = int(bbox[2])
        #         b = int(bbox[3])
        #         cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
        #     img = cv2.resize(img, (1280, 768))
        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)

        # for i, batch in enumerate(self.target_train_loader):
        #     for j in batch:
        #         print(j.shape if isinstance(j, torch.Tensor) else j)
        #     img_weak = batch[0].squeeze(0).numpy().astype(np.uint8)
        #     img_weak = img_weak.copy()
        #     img_strong = batch[0].squeeze(0).numpy().astype(np.uint8)
        #     img_strong = img_strong.copy()
        #     img_weak = cv2.resize(img_weak, (1280, 736))
        #     img_strong = cv2.resize(img_strong, (1280, 736))
        #     cv2.imshow('img_weak', img_weak)
        #     cv2.imshow('img_strong', img_strong)
        #     cv2.waitKey(0)

        # '''
        logger.info("init prefetcher, this might take one minute or less...")
        self.target_prefetcher = DataPrefetcher(self.target_train_loader)
        self.source_prefetcher_1 = DataPrefetcher(self.source_train_loader_1)
        self.source_prefetcher_2 = DataPrefetcher(self.source_train_loader_2)
        # max_iter means iters per epoch
        # self.max_iter = max(len(self.source_train_loader), len(self.target_train_loader))
        self.max_iter = max(len(self.source_train_loader_1), len(self.source_train_loader_2))

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model_1 = DDP(model_1, device_ids=[self.local_rank], broadcast_buffers=False)
            modelTeacher_1 = DDP(modelTeacher_1, device_ids=[self.local_rank], broadcast_buffers=False)
            model_2 = DDP(model_2, device_ids=[self.local_rank], broadcast_buffers=False)
            modelTeacher_2 = DDP(modelTeacher_2, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model_1 = ModelEMA(model_1, 0.9998)
            self.ema_model_1.updates = self.max_iter * self.start_epoch
            self.ema_model_2 = ModelEMA(model_2, 0.9998)
            self.ema_model_2.updates = self.max_iter * self.start_epoch

        self.model_1, self.model_2 = model_1, model_2
        self.modelTeacher_1, self.modelTeacher_2 = modelTeacher_1, modelTeacher_2
        self.model_1.train()
        self.model_2.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed, testdev=True
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))
        # '''

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                max(self.best_ap_1 * 100, self.best_ap_2 * 100)
            )
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.source_train_loader_1.close_mosaic()
            self.source_train_loader_2.close_mosaic()
            self.target_train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model_1.module.head.use_l1 = True
                self.model_2.module.head.use_l1 = True
            else:
                self.model_1.head.use_l1 = True
                self.model_2.head.use_l1 = True

            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch", k=1)
                self.save_ckpt(ckpt_name="last_mosaic_epoch", k=2)

    def after_epoch(self):
        if self.use_model_ema:
            self.ema_model_1.update_attr(self.model_1)
            self.ema_model_2.update_attr(self.model_2)

        self.save_ckpt(ckpt_name="latest", k=1)
        self.save_ckpt(ckpt_name="latest", k=2)

        if (self.epoch + 1) == self.exp.supervise_epoch:
            self.save_ckpt(ckpt_name="last_supervise", k=1)
            self.save_ckpt(ckpt_name="last_supervise", k=2)

        if (self.epoch + 1) % 1 == 0:
            self.save_ckpt(ckpt_name="{}".format(self.epoch + 1), k=1)
            self.save_ckpt(ckpt_name="{}".format(self.epoch + 1), k=2)

        if (self.epoch + 1) < self.exp.supervise_epoch:
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                all_reduce_norm(self.model_1)
                all_reduce_norm(self.model_2)
                self.evaluate_and_save_model()
        else:
            if (self.epoch + 1) % 1 == 0:
                all_reduce_norm(self.model_1)
                all_reduce_norm(self.model_2)
                self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information

        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")

            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )
            for k, v in loss_meter.items():
                if k not in self.iter_loss:
                    self.iter_loss[k] = [v.latest]
                else:
                    self.iter_loss[k].append(v.latest)

            # loss_str = ", "
            # for k, v in loss_meter.items():
            #     if v.latest is not None:
            #         loss_str.join("{}: {:.3f}".format(k, v.latest))
            #     else:
            #         loss_str.join("{}: None".format(k))
            #
            # for k, v in loss_meter.items():
            #     if k not in self.iter_loss:
            #         if v.latest is not None:
            #             self.iter_loss[k] = [v.latest]
            #         else:
            #             self.iter_loss[k] = [0]
            #     else:
            #         if v.latest is not None:
            #             self.iter_loss[k].append(v.latest)
            #         else:
            #             self.iter_loss[k].append(0)

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.source_train_loader_1, self.source_train_loader_2, self.target_train_loader, self.epoch, self.rank,
                self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model, ckpt_file):
        if self.args.resume:
            logger.info("resume training")
            # if self.args.ckpt is None:
            #     ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
            # else:
            #     ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                # ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)
                logger.info("loading {} checkpoint for fine tuning".format(ckpt['start_epoch']))
                ckpt = ckpt["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0
        return model

    def evaluate_and_save_model(self):
        evalmodel = self.ema_model_1.ema if self.use_model_ema else self.model_1
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model_1.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val_1/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val_1/COCOAP50_95", ap50_95, self.epoch + 1)
            for k, v in self.iter_loss.items():
                loss = sum(self.iter_loss[k]) / len(self.iter_loss[k])
                self.tblogger.add_scalar("train/{}".format(k), loss, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        # self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap_1, k=1)
        self.best_ap_1 = max(self.best_ap_1, ap50)

        evalmodel = self.ema_model_2.ema if self.use_model_ema else self.model_2
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model_2.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val_2/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val_2/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        # self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap_2, k=2)
        self.best_ap_2 = max(self.best_ap_2, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, k=1):
        if self.rank == 0:
            if k == 1:
                save_model = self.ema_model_1.ema if self.use_model_ema else self.model_1
            else:
                save_model = self.ema_model_2.ema if self.use_model_ema else self.model_2
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name + '_{}'.format(k) if not update_best_ckpt else '{}'.format(k),
            )
