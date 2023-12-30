# Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video


### 1. Introduction
This is the reserch code of the IEEE Transactions on Geoscience and Remote Sensing 2023 paper "Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video".

[X. Zheng, H. Cui and X. Lu, "Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-11, 2023, Art no. 5626911, doi: 10.1109/TGRS.2023.3336665.](https://ieeexplore.ieee.org/document/10330103)

Satellite videos capture the dynamic changes in a large observed sense, which provides an opportunity to track the object trajectories. However, existing multiple object tracking (MOT) methods require massive video annotations, which is time-consuming and fallible. To alleviate this problem, this article proposes a cross-domain multiple object tracker (CDTrack) to learn knowledge from multiple source domains. First, a cross-domain object detector with multilevel domain alignment is constructed to learn domain-invariant knowledge between remote sensing images and satellite videos. Second, the proposed method adopts a bidirectional teacher–student framework to fuse multiple source domains. Two teacher–student models learn different domain knowledge and teach mutually each other. With mutual learning, the proposed method alleviates the discrepancies between different domains. Finally, a simple weakly supervised Re-IDentification (Re-ID) model is proposed for long-term association. Experimental results on the satellite video datasets demonstrate that the proposed method can achieve great performance without satellite video annotations. The code is available at https://github.com/XiangtaoZheng/CDTrack .

### 2. Start
#### Installing
1. **Create Conda environment.**
          
        conda create -n cdtrack python=3.7
        conda activate cdtrack


2. **Install pytorch.**
          
        conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0


3. **Download CDTrack.**

4. **Install requirements and others.**
          
        pip3 install -r requirements.txt
        python3 setup.py develop
        pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
        pip3 install cython_bbox faiss-cpu faiss-gpu


#### Data 
1. **Download DIOR, DOTA and AIR-MOT datasets and put them as follows:**

&emsp;|- YOUR_DATA/
&emsp;&emsp;&emsp;|- DIOR/
&emsp;&emsp;&emsp;&emsp;&emsp;|- annotations/    
&emsp;&emsp;&emsp;&emsp;&emsp;|- ImageSets/  
&emsp;&emsp;&emsp;&emsp;&emsp;|- trainval/
&emsp;&emsp;&emsp;&emsp;&emsp;|- test/
&emsp;&emsp;&emsp;|- DOTA/
&emsp;&emsp;&emsp;&emsp;&emsp;|- annotations/
&emsp;&emsp;&emsp;&emsp;&emsp;|- train/
&emsp;&emsp;&emsp;&emsp;&emsp;|- val/
&emsp;&emsp;&emsp;|- AIRMOT/
&emsp;&emsp;&emsp;&emsp;&emsp;|- annotations/
&emsp;&emsp;&emsp;&emsp;&emsp;|- gt/
&emsp;&emsp;&emsp;&emsp;&emsp;|- train/
&emsp;&emsp;&emsp;&emsp;&emsp;|- test/


2. **Convert the datasets to COCO format:**
            
        python tools/datasets/convert_dior_to_coco.py
        python tools/datasets/convert_dota_to_coco.py
        python tools/datasets/convert_airmot_to_coco.py


#### Training
            
        python yolox/multi_train.py -expn <experiment_name> -f yolox/exps/example/mot/yolox_s_multi_source.py -d 4 -b 16 --fp16 -c <pretrained_model_1>-<pretrained_model_2>

1. **For training Re-ID model, pseudo labels must be generated as follows:**
            
        python python generate_reid_pseudo_labels.py --input_dir <video_dir> --output_dir <tracking_results> -c <model_path>
        python fast_reid/datasets/generate_pseudo_patches.py --data_path <tracking_results> --save_path <save_path>
        python fast_reid/tools/train_net.py --config-file ./fast_reid/configs/AIRMOT/sbs_R18.yml MODEL.DEVICE "cuda:0"

#### Test

    python run.py --input_dir <video_dir> --output_dir <save_path> -c <model_path> --with-reid --fast-reid-config <reid_config> --fast-reid-weights <reid_model>


### 3. Related work 
If you find the code and dataset useful in your research, please consider citing:

    @article{Zheng2023Multiple,
    author={Zheng, Xiangtao and Cui, Haowen and Lu, Xiaoqiang},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Multiple Source Domain Adaptation for Multiple Object Tracking in Satellite Video}, 
    year={2023},
    volume={61},
    number={},
    pages={1-11},
    doi={10.1109/TGRS.2023.3336665}}
    
#### Acknowledgement
A large part of the codes are borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack),[BoT-SORT](https://github.com/NirAharon/BoT-SORT). Thanks for their excellent works!