# Learning to Generate Language-supervised and Open-vocabulary Scene Graph using Pre-trained Visual-Semantic Space

## Installation and Setup

***Environment.***
This repo requires Pytorch>=1.9 and torchvision.

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo 
pip install transformers 
pip install SceneGraphParser spacy 
python setup.py build develop --user
```

***Pre-trained Visual-Semantic Space.*** Download the pre-trained `GLIP-T` and `GLIP-L` [checkpoints](https://github.com/microsoft/GLIP#model-zoo) into the ``MODEL`` folder. 
(!! GLIP has updated the downloading paths, please find these checkpoints following https://github.com/microsoft/GLIP#model-zoo)
```
mkdir MODEL
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O swin_tiny_patch4_window7_224.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O swin_large_patch4_window12_384_22k.pth
```

## Dataset Preparation

1. Download original datasets
* ``COCO``: Download the original [COCO](https://cocodataset.org/#download) data into ``DATASET/coco`` folder. Refer to [coco_prepare](https://github.com/microsoft/GLIP/blob/main/DATA.md).
* ``Visual Genome (VG)``: Download the original [VG](https://visualgenome.org/) data into ``DATASET/VG150`` folder. Refer to [vg_prepare](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md).

The `DATASET` directory is organized roughly as follows:
```
├─coco
│  ├─annotations
│  ├─train2017
│  ├─val2017
│  └─zero-shot
└─VG150
    ├─VG_100K
    ├─weak_supervisions
    ├─image_data.json
    ├─VG-SGG-dicts-with-attri.json
    ├─region_descriptions.json
    └─VG-SGG-with-attri.h5 
```

Since GLIP pre-training has seen part of VG150 test images, we remove these images and get new VG150 split and write it to `VG-SGG-with-attri.h5`. 
Please refer to [tools/cleaned_split_GLIPunseen.ipynb](tools/cleaned_split_GLIPunseen.ipynb).

2. For language-supervised SGG, we parse scene graph supervision as follows:

```
### Unlocalized VG scene graphs for SGG
python tools/data_preprocess/ground_unlocalized_graph.py
``` 

```
### VG image-caption pairs for language-supervised SGG
python tools/data_preprocess/parse_SG_from_VG_caption.py
```

```
### COCO image-caption pairs for language-supervised SGG
# extract triplets from all related captions, then ground these triplets
python tools/data_preprocess/parse_SG_from_COCO_captionV2.py 
```

Note that, to utilize the advanced scene graph parser from https://nlp.stanford.edu/software/scenegraph-parser.shtml, please refer to [tools/data_preprocess/CocoCaptionParser.java](tools/data_preprocess/CocoCaptionParser.java).


3. Obtain scene graph groundings (for language-supervised SGG)
```
python tools/data_preprocess/parse_SG_from_COCO_caption.py
```
For speeding up, this process supports multi-GPU running by setting params like `--gpu_size 10 --local_rank 1`, 
Next, run `merge_coco_multiGPU_grounding_results.ipynb` to merge multi-GPU results.

## Training & Evaluation

1. Fully supervised SGG
```
# first set NUM_GPUS, e.g., NUM_GPUS=8

# VS$^3$ (Swin-T) -- training
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \

# VS$^3$ (Swin-T) -- evaluation
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/test_grounding_net.py \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml --task_config configs/vg150/finetune.yaml MODEL.WEIGHT $your_checkpoint_file \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} ))

# VS$^3$ (Swin-L) -- training
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_L.yaml MODEL.WEIGHT MODEL/glip_large_model.pth \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
```

2. Language-supervised SGG 

```
# VS$^3$ (Swin-T) -- unlocalized graph
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.VG150_GRAPH_GROUNDING_FILE "DATASET/VG150/weak_supervisions/language_supervised-unlocalized_graph-grounding=glipL.json"

# VS$^3$ (Swin-T) -- VG caption
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.TRAIN "('vgcaption_scene_graph',)"

# VS$^3$ (Swin-T) -- COCO caption
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.TRAIN "('cococaption_scene_graph',)"
```
If the `Data Preparation` is correctly done, we will get data files needed in the above runs.

3. Open-vocabulary SGG

```
# VS$^3$ (Swin-T) -- Fully supervised with manual annotations
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.VG150_OPEN_VOCAB_MODE True

# VS$^3$ (Swin-T) -- Language-supervised with VG caption
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py \
    --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    SOLVER.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) TEST.IMS_PER_BATCH $(( 1 * ${NUM_GPUS} )) \
    MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False DATASETS.TRAIN "('vgcaption_scene_graph',)" DATASETS.VG150_OPEN_VOCAB_MODE True
```

## Acknowledgement

This repo is based on [GLIP](https://github.com/microsoft/GLIP), [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS). Thanks for their contribution.
