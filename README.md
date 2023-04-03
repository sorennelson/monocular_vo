# Monocular Visual Odometry

> A comparison of Unsupervised Deep Learning and Classical Geometric methods for monocular ego-motion estimation on KITTI Odometry.

## Deep Unsupervised

### SfMLearner

Unsupervised method to jointly train pose and depth estimation models with a novel view synthesis loss, proposed by Zhou et al. in [Unsupervised Learning of Depth and Ego-Motion from Video.](https://arxiv.org/abs/1704.07813) 

**Modifications:**
- Grayscale as opposed to color videos.
- No Batch Norm as in the post-publication updates in the [official codebase.](https://github.com/tinghuiz/SfMLearner/tree/master)
- No Explainability mask as in the post-publication updates in the [official codebase.](https://github.com/tinghuiz/SfMLearner/tree/master) This can be toggled on by setting the `lambda_e` arg to non-zero.
- A single-scale loss on the full-scale predictions as opposed to the multi-scale depth/explainability predictions performs slightly better (and trains faster), similar to what Bian et al. found in [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video.](https://proceedings.neurips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf) This can be adjusted by setting the `n_scales` arg (up to 4 as in the original paper).

**Data:**
1. Download [KITTI Odometry grayscale dataset and poses.](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
2. Split train/val data and setup snippets.
    ```
    python data/prepare_train_data.py --dataset_dir path/to/kitti/ --dataset_name kitti_odom --dump_root path/to/output/ --seq_length [3,5]
    ```
3. Setup test snippets.
    ```
    python data/prepare_test_data.py --dataset_dir path/to/kitti/ --dataset_name kitti_odom --dump_root path/to/output/ --seq_length [3,5]
    ```
4. Compress and set up each train/val/test split for [MosaicML Streaming Dataset](https://github.com/mosaicml/streaming) which can then be used locally or remotely. 
    ```
    python data/prepare_stream_data.py --dataset_dir path/to/dump_root/from/above/ --split [train,val,test] --dump_root path/to/output/
    ```

**Training:**
```
python main.py --data_path path/to/streaming/ds --job_id [job_id] --epochs 25 --batch_size 4 --skip --n_src [2,4] --decay 0. --lambda_s 0.5 --lambda_e 0. --n_scales 1
```

**Evaluation:**
```
python main.py --data_path path/to/streaming/ds --job_id [job_id] --epochs 25 --batch_size 1 --skip --n_src [2,4] --decay 0. --lambda_s 0.5 --lambda_e 0. --n_scales 1 --evaluate
```


## Classical

### Manual

A manual baseline implementation of classical VO (using OpenCV only for feature extraction). I first match the 300 closest keypoints across image pairs using ORB features. I then estimate the Essential matrix with Normalized 8-point algorithm and Ransac, and finally extract the pose with the highest cheirality.

**Running:**
```
python classical_vo.py --version manual --seq_path path/to/kitti/sequence pose_txt_path path/to/kitti/seq/pose.txt
```

### OpenCV

A fully OpenCV implementation of classical VO. I track features using FAST feature detector and Lucas Kanade, re-estimating new features if < 2000 keypoints remain in the current frame. I then estimate the Essential matrix with Nister's 5-Point and Ransac, and finally extract the pose.

**Running:** 
```
python classical_vo.py --version opencv --seq_path path/to/kitti/sequence pose_txt_path path/to/kitti/seq/pose.txt
```

## Results

Average Absolute Trajectory Error (ATE) on the KITTI Odometry split (sequences 09-10) averaged over 3 runs. All models are trained on sequences 00-08. For a fair timing comparison, inference for all methods is run on my 2019 16" MacBook Pro (2.6 GHz 6-Core i7).

Method | ATE (3 frame)
--- | --- 
SfMLearner (Multi-Scale) | 0.0088
SfMLearner (Single-Scale with Explainability) | 0.0093 
SfMLearner (Single-Scale) | 0.0084


Method | ATE (5 frame) | ATE (3 frame) | Total Inference Time (Seconds)
--- | --- | --- | --- 
Manual (ORB + 8-Point) | 0.1612 | 0.1186 | 796
OpenCV (FAST + LK + Nister's 5-Point) | 0.0464 | 0.0357 | 931 
SfMLearner | 0.0158 | 0.0084 | 32


## Unsupervised Deep Monocular VO Resources

- [Unsupervised Learning of Depth and Ego-Motion from Video](https://arxiv.org/abs/1704.07813)
- [Official Codebase](https://github.com/tinghuiz/SfMLearner/tree/master)
- [PyTorch reimplementation](https://github.com/ClementPinard/SfmLearner-Pytorch)
- [Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video](https://proceedings.neurips.cc/paper/2019/file/6364d3f0f495b6ab9dcf8d3b5c6e0b01-Paper.pdf)
- [Towards Better Generalization: Joint Depth-Pose Learning without PoseNet](https://arxiv.org/abs/2004.01314)
- [Awesome Deep Visual Odometry](https://github.com/hassaanhashmi/awesome-deep-visual-odometry)







