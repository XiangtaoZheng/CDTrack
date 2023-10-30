# Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video


## Introduction
This is the reserch code of the IEEE Transactions on Geoscience and Remote Sensing 2023 paper.

X. Zheng, H. Cui and X. Lu, "Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video," IEEE Transactions Geoscience and Remote Sensing, 2023.

In this code, we explored cross-domain multiple object tracking learned knowledge from multiple source domains in satellite videos.

## Usage

### Installation
**Step 1.** Create Conda environment.
```shell
conda create -n cdtrack python=3.7
conda activate cdtrack
```

**Step 2.** Install pytorch.
```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0
```

**Step 3.** Download CDTrack.

**Step 4.** Install requirements and others.
```shell
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip3 install cython_bbox faiss-cpu faiss-gpu
```


### Data 
**Step 1.** Download DIOR, DOTA and AIR-MOT datasets and put them as follows:
```shell script
# YOUR_DATA/
#   DIOR/
#     annotations/    
#     ImageSets/  
#     trainval/
#     test/
#   DOTA/
#     annotations/
#     train/
#     val/
#   AIRMOT/
#     annotations/
#     gt/
#     train/
#     test/
```
**Step 2.** Convert the datasets to COCO format:
```shell script
python tools/datasets/convert_dior_to_coco.py
python tools/datasets/convert_dota_to_coco.py
python tools/datasets/convert_airmot_to_coco.py
```



### Training
```shell script
python yolox/multi_train.py -expn <experiment_name> -f yolox/exps/example/mot/yolox_s_multi_source.py -d 4 -b 16 --fp16 -c <pretrained_model_1>-<pretrained_model_2>
```
- For training Re-ID model, pseudo labels must be generated as follows:
```shell script
python python generate_reid_pseudo_labels.py --input_dir <video_dir> --output_dir <tracking_results> -c <model_path>
python fast_reid/datasets/generate_pseudo_patches.py --data_path <tracking_results> --save_path <save_path>
python fast_reid/tools/train_net.py --config-file ./fast_reid/configs/AIRMOT/sbs_R18.yml MODEL.DEVICE "cuda:0"
```



### Test
```shell script
python run.py --input_dir <video_dir> --output_dir <save_path> -c <model_path> --with-reid --fast-reid-config <reid_config> --fast-reid-weights <reid_model>
```

## Cite
```
@article{zheng2023multi,
 title={Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video},
 author={Zheng, Xiangtao and Cui, Haowen and Lu, Xiaoqiang},
 journal={IEEE Transactions on Geoscience and Remote Sensing},
 year={2023}
 }
```

## Acknowledgement
A large part of the codes are borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack),[BoT-SORT](https://github.com/NirAharon/BoT-SORT). Thanks for their excellent works!