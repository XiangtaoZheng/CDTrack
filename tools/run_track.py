from loguru import logger

import cv2
import torch
import tqdm
import numpy as np

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking

from tracker.byte_tracker import BYTETracker
from tracker.tracking_utils.timer import Timer
from tracker.basetrack import BaseTrack
from tracker.mc_bot_sort import BoTSORT

import argparse
import os
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument('input_dir', default='')
    parser.add_argument('output_dir', default='')
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--tracker", default="bot", help="tracker"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # "--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/palace.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=int, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=1, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # BOTSORT
    parser.add_argument("--cmc-method", default="orb", type=str,
                        help="cmc method: files (Vidstab GMC) | orb | ecc |none")
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"logs/viso/sbs_S50/config.yaml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"logs/viso/sbs_S50/model_final.pth",
                        type=str, help="reid config file path")

    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def byte_track(predictor, path, args, exp):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=10)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in tqdm.tqdm(files):
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        outputs, img_info = predictor.inference(image_name, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_category = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tcategory = t.category
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_category.append(tcategory)

            timer.toc()
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores, online_category))
        frame_id += 1
    basetrack = BaseTrack()
    basetrack.reset()
    return results


def bot_sort(predictor, path, args, exp):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = BoTSORT(args, frame_rate=10)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in tqdm.tqdm(files):
        outputs, img_info = predictor.inference(image_name, timer)
        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
        if outputs[0] is not None:
            outputs = outputs[0]
            detections = outputs[:, :7]
            detections[:, :4] /= scale
            online_targets = tracker.update(detections, img_info['raw_img'])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_category = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tcategory = t.cls
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_category.append(tcategory)

            timer.toc()
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores, online_category))
        frame_id += 1
    basetrack = BaseTrack()
    basetrack.reset()
    return results


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    if isinstance(model, tuple):
        model = model[0]
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        # model.load_state_dict(ckpt)
        logger.info("loaded {} checkpoint done.".format(ckpt['start_epoch']))
        # torch.save(model.state_dict(), ckpt_file[:-4], _use_new_zipfile_serialization=False)
    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, None, None, args.device, args.fp16)
    return predictor


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
