
Multi-Stage Synergistic Aggregation Network for Remote Sensing Visual Grounding
========
This is the official repository of 'Multi-Stage Synergistic Aggregation Network for Remote Sensing Visual Grounding'. 

## Introduction
This project contains a method that leverages cross-attention and query channel broadcasting as two fusion kernels involving both queries in the Multi-Stage Synergistic Aggregation Module (MSAM) with Swin transformer and GPT-like generative manner.
<p align="center">
  <img src="./docs/framework.drawio.svg" />
</p>
<p align="center">
  <img src="./docs/heatmap.drawio.svg" />
</p>



## Model Repository

The best models and ablation study models are available in [Google Drive](https://drive.google.com/drive/folders/1_YwhNNxjfhMg3ikMhjyDhi1sLUPKASWm?usp=share_link). The ablation study code branches will be gradually open-sourced in this repo.


## Installation
### Docker installation - RECOMMENDED (storage 7.6 GB)
```
docker pull waynamigo/msam:py38t1.9
```
### Conda/pip installation
1. install pytorch and torchvision
```
conda install pytorch==1.9.1 torchvision==0.10.1 -c pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
2. install other requirements
```
pip install requirements.txt
```
3. openmin install mmmcv(v1.7.1)
```
mim install mmcv-full==1.7.1
```
4. if there's unexpected error like FormatCode or sth:
```
try downgrade or upgrade deps.

eg. TypeError: FormatCode() got an unexpected keyword argument 'verify'
downgrade yapf==0.40.2 to 0.40.1
```
## Preparation
1. The preparing the [DIOR-RSVG](https://github.com/ZhanYang-nwpu/RSVG-pytorch/tree/main/DIOR_RSVG) dataset. 
2. Run `bash scripts/xml2instances.py` to generate available format for our dataset preparation. The prepared data dir tree is this:
```
└── annotations(origion xml dataset from DIOR-RSVG)
│   └── rsvgd
│        ├── instances.json
│        ├── ix_to_token.pkl
│        ├── token_to_ix.pkl
│        └── word_emb.npz
├── images
│   └── rsvgd
└── weights
    ├── darknet.weights
    ├── yolov3.weights
    └── detr-r50.pth
```
## Training

The following is an example of model training on the RefCOCOg dataset.
```
python tools/train.py configs/msam/detection/msam_rsvgd.py --cfg-options ema=True
```
We train the model on 3090 with a total batch size of 16 for 80 epochs, occupying a minimum of 18GB of VRAM.
## Evaluation
Run the following script to evaluate the trained model with a single GPU.
`python tools/test.py <config-path> --load-from <model-path>`
```
python tools/test.py models/20230520_120410_qb_ca_mixed/20230520_120410_qb_ca_mixed.py --load-from work_dir/20230520_120410_qb_ca_mixed/latest.pth 
```

## Acknowledgement
Part of our code is based on the previous works [Swin](https://github.com/microsoft/Swin-Transformer) 
and [SeqTR](https://github.com/seanzhuh/SeqTR)
and [SKNet](https://github.com/implus/SKNet)
