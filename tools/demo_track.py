from loguru import logger

import sys
for path in sys.path:
    if 'Python' in path:
        sys.path.remove(path)

import cv2

import torch

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.visualize import plot_tracking

# from tracker.bot_sort import BoTSORT
from tracker.mc_bot_sort import BoTSORT
from tracker.byte_tracker import BYTETracker
from tracker.tracking_utils.timer import Timer
from tracker.strongsort.strongsort import Tracker
from tracker.strongsort.nn_matching import NearestNeighborDistanceMetric

import argparse
import os
import time

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--tracker", default="byte", help="tracker"
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
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument("--plane-config", dest="plane_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--plane-weights", dest="plane_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument("--ship-config", dest="ship_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--ship-weights", dest="ship_weights", default=r"pretrained/mot17_sbs_S50.pth",
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


def byte_track(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=10)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
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
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                      scores=online_scores, fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']
            cv2.putText(online_im, 'frame: %d fps: %.2f num: %d' % (frame_id, 1. / timer.average_time, 0),
                        (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        frame_id += 1

        # vis_track(online_im)
        # cv2.waitKey(1)

        video = image_name.split('/')[-3]
        img_name = image_name.split('/')[-1]
        # print(video, img_name)
        save_dir = 'track_results/byte_source_{}/'.format(video)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, img_name)
        online_im = cv2.resize(online_im, (1024, 1024))
        cv2.imwrite(save_path, online_im)


def bot_sort(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    tracker = BoTSORT(args, frame_rate=10)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        frame_id += 1
        if frame_id % 1 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
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
            results.append((frame_id, online_tlwhs, online_ids, online_scores, online_category))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id,
                                      scores=online_scores, fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']
            cv2.putText(online_im, 'frame: %d fps: %.2f num: %d' % (frame_id, 1. / timer.average_time, 0),
                        (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        vis_track(online_im)
        cv2.waitKey(1)

def strong_sort(predictor, path, args, exp):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    metric = NearestNeighborDistanceMetric(
        'cosine',
        0.4,
        1
    )
    tracker = Tracker(metric, args)
    timer = Timer()
    frame_id = 0
    results = []
    for image_name in files:
        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        outputs, img_info = predictor.inference(image_name, timer)
        scale = min(exp.test_size[0] / float(img_info['height']), exp.test_size[1] / float(img_info['width']))
        if outputs[0] is not None:
            outputs = outputs[0]
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            video = image_name.split('/')[-3]
            tracker.camera_update(video, frame_id + 1)
            tracker.predict()
            tracker.update(detections, img_info['raw_img'])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_category = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                online_tlwhs.append(track.to_tlwh())
                online_ids.append(track.track_id)
                online_scores.append(1)
                online_category.append(0)
            results.append((frame_id + 1, online_tlwhs, online_ids, online_scores, online_category))
            timer.toc()
            results.append((frame_id, online_tlwhs, online_ids, online_scores, online_category))
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id,
                                      scores=online_scores, fps=1. / timer.average_time)
        else:
            timer.toc()
            online_im = img_info['raw_img']
            cv2.putText(online_im, 'frame: %d fps: %.2f num: %d' % (frame_id, 1. / timer.average_time, 0),
                        (0, int(15 * 2)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

        vis_track(online_im)
        cv2.waitKey(1)
        frame_id += 1



def vis_track(online_im):
    online_im = cv2.resize(online_im, (1280, 768))
    cv2.imshow('{}'.format('track'), online_im)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    if args.save_result:
        vis_folder = os.path.join(file_name, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = None

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
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        logger.info("finish loading epoch {}".format(ckpt['start_epoch']))
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        if args.tracker == 'byte':
            byte_track(predictor, vis_folder, args.path, current_time, args.save_result)
        elif args.tracker == 'bot':
            bot_sort(predictor, vis_folder, args.path, current_time, args.save_result)
        elif args.tracker == 'strong':
            strong_sort(predictor, args.path, args, exp)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
