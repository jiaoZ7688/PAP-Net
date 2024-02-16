import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np
import cv2
from PIL import Image
from skimage import measure

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes, polygons_to_bitmask
from fvcore.common.file_io import PathManager, file_lock


from detectron2.data import MetadataCatalog, DatasetCatalog
import sys

from PIL import Image, ImageDraw, ImageFilter
import imantics
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from copy import deepcopy


# from aistron import datasets # register data

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_coco_json"]

def get_binary_mask_from_polygon(segs, height, width):
    polygons = []
    for seg in segs:
        polygons.append(np.array(seg).reshape((int(len(seg)/2), 2)))
    formatted_polygons = []

    for polygon in polygons:
        formatted_polygon = []
        for p in polygon:
            formatted_polygon.append((p[0], p[1]))

        formatted_polygons.append(formatted_polygon)
    
    img = Image.new('L', (width, height), 0)
    for formatted_polygon in formatted_polygons:
        ImageDraw.Draw(img).polygon(formatted_polygon, outline=1, fill=1)
    mask = np.array(img)
    return mask

def from_rle_str_to_plg(encode_str):
    binary_mask = maskUtils.decode(encode_str)
    polygons = imantics.Mask(binary_mask).polygons()

    return [l.tolist() for l in polygons] 

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        print('meta:', meta)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = ['foreground']

        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = {1: 0}
        
    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = [ "id", "category_id", "iscrowd", "bbox", "area", "segmentation", "i_segmentation", "bg_object_segmentation", "occlude_rate"] + (extra_annotation_keys or [])

    for jdex, (img_dict, anno_dict_list) in enumerate(imgs_anns):
        record = {}
        #record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["file_name"] = img_dict["file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        print('file name:', jdex, ':', record["file_name"])
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0

            obj = {key: anno[key] for key in ann_keys if key in anno}
            # try:
            #     obj['i_bbox'] = anno['i_bbox']
            # except:
            #     obj['i_bbox'] = anno['bbox']

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = 0
            objs.append(obj)

        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts # [ img1_ann, img2_ann........],   img_ann = record: dict


def convert_to_coco_dict(dataset_dicts, dataset_name):
    """
    Convert a dataset in detectron2's standard format into COCO json format

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name:
            name of the source dataset
            must be registered in DatastCatalog and in detectron2's standard format
    Returns
        coco_dict: serializable dict in COCO json format
    """

    #dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()}
        print('metadata.thing_dataset_id_to_contiguous_id:', metadata.thing_dataset_id_to_contiguous_id)
        print('reverse_id_mapping:', reverse_id_mapping)
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
        print('continue')
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    print('categories:', categories)
    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            coco_annotation["id"] = annotation["id"]
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = annotation["bbox"] # convert to xyxy format
            coco_annotation["area"] = annotation["area"]
            coco_annotation["iscrowd"] = annotation["iscrowd"]
            coco_annotation["category_id"] = reverse_id_mapper(annotation["category_id"])

            coco_annotation["occlude_rate"] = annotation["occlude_rate"]

            if "segmentation" in annotation:
                coco_annotation["segmentation"] = annotation["segmentation"]
            if "bg_object_segmentation" in annotation:
                coco_annotation["bg_object_segmentation"] = annotation["bg_object_segmentation"]
            if "i_segmentation" in annotation:
                coco_annotation["i_segmentation"] = annotation["i_segmentation"]


            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict


if __name__ == "__main__":
    """
    Test the COCO json dataset loader.

    Usage:
        python -m detectron2.data.datasets.coco \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "coco_2014_minival_100", or other
        pre-registered ones
    """

    from detectron2.data.datasets import register_coco_instances
    

    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    
    # data is already registered in builtin.py

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_coco_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    coco_dict = convert_to_coco_dict(dicts, sys.argv[3])

    output_file = os.path.splitext(sys.argv[1].split('/')[-1])[0] + '_cls.json'
    ann_path = os.path.dirname(sys.argv[1])
    output_file = os.path.join(ann_path, output_file)

    with open(output_file, "w") as json_file:
        logger.info(f"Caching annotations in COCO format: {output_file}")
        json.dump(coco_dict, json_file)