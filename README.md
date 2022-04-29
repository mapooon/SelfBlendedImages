# Detecting Deepfakes with Self-Blended Images
![Overview](overview.png)  
The official PyTorch implementation for the following paper: 
> [**Detecting Deepfakes with Self-Blended Images**](),  
> Kaede Shiohara and Toshihiko Yamasaki,  
> *CVPR 2022 Oral*

***2022.4.29: After discussions with the university's TLO, it was decided that the code would be licensed (free for research purposes only, with a fee for commercial use). Therefore it takes even more time to create the license. Thank you for your understanding.***

***2022.4.19: Due to some circumstances at our university, the code will be released after 2022.4.26. Please stay tuned.***

<!-- 
# Recomended Development Environment
* GPU: NVIDIA A100
* CUDA: 11.1
* Docker: 20.10.8


# Setup
## Dataset
Download datasets and place them in `./data/` folder.
For example, download [Celeb-DF-v2](https://github.com/yuezunli/celeb-deepfakeforensics) and place it:
```
.
└── data
    └── Celeb-DF-v2
        ├── Celeb-real
        │   └── videos
        │       └── *.mp4
        ├── Celeb-synthesis
        │   └── videos
        │       └── *.mp4
        ├── Youtube-real
        │   └── videos
        │       └── *.mp4
        └── List_of_testing_videos.txt
```
For other datasets, please refer to `./data/datasets.md` .

## Landmark Detector
We use 81 landmarks detector in training.  
Download [here](https://github.com/codeniko/shape_predictor_81_face_landmarks) and place it in `./src/preprocess/` folder.  

## Pretrained model
We provide pretrained EfficientNet-B4.  
Download [here]() and place it in `./weights/` folder.

## Execute docker
1. Replace the absolute path to this repository in `./exec.sh` .
2. Run the shell scripts:
```bash
bash build.sh
bash exec.sh
```


# Test
For example, run the inference on Celeb-DF-v2:
```bash
CUDA_VISIBLE_DEVICES=* python3 src/inference/inference_dataset.py \
-w weights/efnb4_sbi.tar \
-d CDF
```
The result will be displayed.

We also provide inference code for a single video:
```bash
CUDA_VISIBLE_DEVICES=* python3 src/inference/inference_video.py \
-w weights/efnb4_sbi.tar \
-i /path/to/video.mp4
```
and for an image:
```bash
CUDA_VISIBLE_DEVICES=* python3 src/inference/inference_image.py \
-w weights/efnb4_sbi.tar \
-i /path/to/image.png
```

# Training
1. Download [FF++](https://github.com/ondyari/FaceForensics) real videos and place them in `./data/` folder:
```
.
└── data
    └── FaceForensics++
        ├── original_sequences
        │   └── youtube
        │       └── raw
        │           └── videos
        │               └── *.mp4
        ├── train.json
        ├── val.json
        └── test.json
```
2. Run the two codes to extract the video frames, landmarks, and bboxes: (take about 4h and 1.5h, respectively)
```bash
python3 src/preprocess/crop_dlib_ff.py -d Original
CUDA_VISIBLE_DEVICES=* python3 src/preprocess/crop_retina_ff.py -d Original
```

3. Run the training: (take about 13h)
```bash
CUDA_VISIBLE_DEVICES=* python3 src/train_sbi.py \
src/configs/sbi/efnb4.json \
-n sbi
```
Top five checkpoints will be saved in `./output/` folder. As descrived in our paper, we use a latest one for evaluations. -->

# Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@misc{shiohara2022detecting,
      title={Detecting Deepfakes with Self-Blended Images}, 
      author={Kaede Shiohara and Toshihiko Yamasaki},
      year={2022},
      eprint={2204.08376},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```