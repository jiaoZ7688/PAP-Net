# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
import pickle

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


D2SA_CATEGORIES = [
     {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': '1'},
     {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': '2'},
     {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': '3'},
     {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': '4'},
     {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': '5'},
     {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': '6'},
     {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': '7'},
     {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': '8'},
     {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': '9'},
     {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': '10'},
     {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': '11'},
     {"color": [147, 186, 208], "isthing": 1, "id": 12, "name": "12"},
     {'color': [220, 220, 0], 'isthing': 1, 'id': 13, 'name': '13'},
     {'color': [175, 116, 175], 'isthing': 1, 'id': 14, 'name': '14'},
     {'color': [250, 0, 30], 'isthing': 1, 'id': 15, 'name': '15'},
     {'color': [165, 42, 42], 'isthing': 1, 'id': 16, 'name': '16'},
     {'color': [255, 77, 255], 'isthing': 1, 'id': 17, 'name': '17'},
     {'color': [0, 226, 252], 'isthing': 1, 'id': 18, 'name': '18'},
     {'color': [182, 182, 255], 'isthing': 1, 'id': 19, 'name': '19'},
     {'color': [0, 82, 0], 'isthing': 1, 'id': 20, 'name': '20'},
     {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': '21'},
     {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': '22'},
     {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': '23'},
     {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': '24'},
     {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': '25'},
     {"color": [153, 69, 1], "isthing": 1, "id": 26, "name": "26"},
     {'color': [255, 179, 240], 'isthing': 1, 'id': 27, 'name': '27'},
     {'color': [0, 125, 92], 'isthing': 1, 'id': 28, 'name': '28'},
     {"color": [3, 95, 161], "isthing": 1, "id": 29, "name": "29"},
     {"color": [163, 255, 0], "isthing": 1, "id": 30, "name": "30"},
     {'color': [209, 0, 151], 'isthing': 1, 'id': 31, 'name': '31'},
     {'color': [188, 208, 182], 'isthing': 1, 'id': 32, 'name': '32'},
     {'color': [0, 220, 176], 'isthing': 1, 'id': 33, 'name': '33'},
     {'color': [255, 99, 164], 'isthing': 1, 'id': 34, 'name': '34'},
     {'color': [92, 0, 73], 'isthing': 1, 'id': 35, 'name': '35'},
     {'color': [133, 129, 255], 'isthing': 1, 'id': 36, 'name': '36'},
     {'color': [78, 180, 255], 'isthing': 1, 'id': 37, 'name': '37'},
     {'color': [0, 228, 0], 'isthing': 1, 'id': 38, 'name': '38'},
     {'color': [174, 255, 243], 'isthing': 1, 'id': 39, 'name': '39'},
     {'color': [45, 89, 255], 'isthing': 1, 'id': 40, 'name': '40'},
     {'color': [134, 134, 103], 'isthing': 1, 'id': 41, 'name': '41'},
     {'color': [145, 148, 174], 'isthing': 1, 'id': 42, 'name': '42'},
     {'color': [255, 208, 186], 'isthing': 1, 'id': 43, 'name': '43'},
     {'color': [197, 226, 255], 'isthing': 1, 'id': 44, 'name': '44'},
     {"color": [119, 0, 170], "isthing": 1, "id": 45, "name": "45"},
     {'color': [171, 134, 1], 'isthing': 1, 'id': 46, 'name': '46'},
     {'color': [109, 63, 54], 'isthing': 1, 'id': 47, 'name': '47'},
     {'color': [207, 138, 255], 'isthing': 1, 'id': 48, 'name': '48'},
     {'color': [151, 0, 95], 'isthing': 1, 'id': 49, 'name': '49'},
     {'color': [9, 80, 61], 'isthing': 1, 'id': 50, 'name': '50'},
     {'color': [84, 105, 51], 'isthing': 1, 'id': 51, 'name': '51'},
     {'color': [74, 65, 105], 'isthing': 1, 'id': 52, 'name': '52'},
     {'color': [166, 196, 102], 'isthing': 1, 'id': 53, 'name': '53'},
     {'color': [208, 195, 210], 'isthing': 1, 'id': 54, 'name': '54'},
     {'color': [255, 109, 65], 'isthing': 1, 'id': 55, 'name': '55'},
     {'color': [0, 143, 149], 'isthing': 1, 'id': 56, 'name': '56'},
     {'color': [179, 0, 194], 'isthing': 1, 'id': 57, 'name': '57'},
     {'color': [209, 99, 106], 'isthing': 1, 'id': 58, 'name': '58'},
     {'color': [5, 121, 0], 'isthing': 1, 'id': 59, 'name': '59'},
     {'color': [227, 255, 205], 'isthing': 1, 'id': 60, 'name': '60'},
]


KINS_CATEGORIES = [
     {'color': [220, 20, 60], 'supercategory': "Living thing", 'id': 1, 'name': 'cyclist'},
     {'color': [119, 11, 32], 'supercategory': "Living thing", 'id': 2, 'name': 'pedestrian'},
     {'color': [133, 129, 255], 'supercategory': "Living thing", 'id': 3, 'name': 'rider'},
     {'color': [151, 0, 95], 'supercategory': "vehicles", 'id': 4, 'name': 'car'},
     {'color': [0, 0, 230], 'supercategory': "vehicles", 'id': 5, 'name': 'tram'},
     {'color': [106, 0, 228], 'supercategory': "vehicles", 'id': 6, 'name': 'truck'},
     {'color': [0, 60, 100], 'supercategory': "vehicles", 'id': 7, 'name': 'van'},
     {'color': [227, 255, 205], 'supercategory': "vehicles", 'id': 8, 'name': 'misc'},
]


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

def register_kins():
    register_coco_instances("kins_dataset_train", {}, 
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/train_amodal.json", 
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/train_imgs/image_2"
    )
    # register_coco_instances("kins_dataset_train", {}, 
    #     "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_amodal_cs.json", 
    #     "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_imgs/image_2"
    # )
    
    meta = MetadataCatalog.get('kins_dataset_train')
    meta.thing_classes =['cyclist', 'pedestrian', 'car', 'tram',
                            'truck', 'van', 'misc']                   

    register_coco_instances("kins_dataset_val", {}, 
        # "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_amodal_cs.json", 
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_amodal.json", 
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_imgs/image_2"
    )

    meta = MetadataCatalog.get('kins_dataset_val')
    meta.thing_classes =['cyclist', 'pedestrian',  'car', 'tram',
                            'truck', 'van', 'misc']           

# def register_kins():
#     register_coco_instances("kins_dataset_train", {}, 
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/kins_cls/train_cls_amodal.json", 
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/train_imgs/image_2"
#     )
    
#     meta = MetadataCatalog.get('kins_dataset_train')
#     meta.thing_classes =['foreground']                     

#     register_coco_instances("kins_dataset_val", {}, 
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/kins_cls/test_cls_amodal.json", 
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_imgs/image_2"
#     )

#     meta = MetadataCatalog.get('kins_dataset_val')
#     meta.thing_classes =['foreground']       


def register_COCOA():
    # train data
    register_coco_instances(
        "cocoa_cls_train", {},
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_train2014_with_classes_amodal.json",
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/coco2014/train2014"
    )
    meta = MetadataCatalog.get("cocoa_cls_train")
    cocoa_cat_list = None
    with open('/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/cocoa_cat_list', 'rb') as fp:
        cocoa_cat_list = pickle.load(fp)
    meta.thing_classes = cocoa_cat_list

    # val data
    register_coco_instances(
        "cocoa_cls_val", {},
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_val2014_with_classes_amodal.json",
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/coco2014/val2014"
    )
    meta = MetadataCatalog.get("cocoa_cls_val")
    cocoa_cat_list = None
    with open('/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/cocoa_cat_list', 'rb') as fp:
        cocoa_cat_list = pickle.load(fp)
    meta.thing_classes = cocoa_cat_list

# def register_COCOA():
#     # train data
#     register_coco_instances(
#         "cocoa_cls_train", {},
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_train2014_with_classes_amodal.json",
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/coco2014/train2014"
#     )
#     meta = MetadataCatalog.get("cocoa_cls_train")
#     meta.thing_classes =['foreground']     

#     # val data
#     register_coco_instances(
#         "cocoa_cls_val", {},
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_val2014_with_classes_amodal.json",
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/coco2014/val2014"
#     )
#     meta = MetadataCatalog.get("cocoa_cls_val")
#     meta.thing_classes =['foreground']     


def register_D2SA():
    # train data
    register_coco_instances(
        "d2sa_train", {},
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_training_rot0_amodal.json",
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images",
    )
    meta = MetadataCatalog.get("d2sa_train")
    d2sa_cat_list = None
    with open('/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2sa_cat_list', 'rb') as fp:
        d2sa_cat_list = pickle.load(fp)
    meta.thing_classes = d2sa_cat_list
    # meta.thing_colors = [k["color"] for k in D2SA_CATEGORIES if k["isthing"] == 1] 

    # augmented train data
    register_coco_instances(
        "d2sa_train_aug", {},
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_augmented_amodal.json",
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images",
    )
    meta = MetadataCatalog.get("d2sa_train_aug")
    d2sa_cat_list = None
    with open('/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2sa_cat_list', 'rb') as fp:
        d2sa_cat_list = pickle.load(fp)
    meta.thing_classes = d2sa_cat_list
    # meta.thing_colors = [k["color"] for k in D2SA_CATEGORIES if k["isthing"] == 1] 

    # val data
    register_coco_instances(
        "d2sa_val", {},
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_validation_amodal.json",
        "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images",
    )
    meta = MetadataCatalog.get("d2sa_val")
    d2sa_cat_list = None
    with open('/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2sa_cat_list', 'rb') as fp:
        d2sa_cat_list = pickle.load(fp)
    meta.thing_classes = d2sa_cat_list 
    # meta.thing_colors = [k["color"] for k in D2SA_CATEGORIES if k["isthing"] == 1] 


# def register_D2SA():
#     # train data
#     register_coco_instances(
#         "d2sa_train", {},
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_training_rot0_amodal.json",
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images",
#     )
#     meta = MetadataCatalog.get("d2sa_train")
#     meta.thing_classes =['foreground']     

#     # augmented train data
#     register_coco_instances(
#         "d2sa_train_aug", {},
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_augmented_amodal.json",
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images",
#     )
#     meta = MetadataCatalog.get("d2sa_train_aug")
#     meta.thing_classes =['foreground']     

#     # val data
#     register_coco_instances(
#         "d2sa_val", {},
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_validation_amodal.json",
#         "/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images",
#     )
#     meta = MetadataCatalog.get("d2sa_val")
#     meta.thing_classes =['foreground']     


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)
    register_kins()
    register_COCOA()
    register_D2SA()
