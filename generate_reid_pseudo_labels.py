from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tqdm
from tools.run_track import make_parser
from tools.run_track import main
from tools.run_track import byte_track, bot_sort
from yolox.exp import get_exp

args = make_parser().parse_args()
args.demo = 'image'
args.exp_file = 'yolox/exps/example/mot/yolox_s.py'

args.fuse = True
args.fp16 = True
exp = get_exp(args.exp_file, args.name)
input_dir = args.input_dir
output_dir = args.output_dir
predictor = main(exp, args)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


start = time.time()
for dir in tqdm.tqdm(sorted(os.listdir(input_dir))):
    video_path = os.path.join(input_dir, dir + '/img/')
    results = byte_track(predictor, video_path, args, exp)
    output_path = os.path.join(output_dir, dir + '.txt')
    if os.path.exists(output_path):
        os.remove(output_path)
    output_file = open(output_path, 'w')
    for frame in range(len(results)):
        frame_id = results[frame][0]
        num_objs = len(results[frame][1])
        for obj in range(num_objs):
            track_id = results[frame][2][obj]
            bbox = results[frame][1][obj]
            score = results[frame][3][obj]
            category_id = int(results[frame][4][obj]) + 1
            result = '{},{},{},{},{},{},{},{},1'.format(frame_id, track_id, bbox[0], bbox[1], bbox[2],
                                                       bbox[3], score, category_id)
            result += '' if frame + 1 == len(results) and obj + 1 == num_objs else '\n'
            output_file.write(result)
end = time.time()
seconds = end - start
m, s = divmod(seconds, 60)
h, m = divmod(m, 60)
print("total: %02d:%02d:%02d" % (h, m, s))
