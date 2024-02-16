from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import imantics
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tools.vis_gt import IMG_DIR
from skimage.transform import resize
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.structures import Boxes
import torch
import os
from os.path import join
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.structures import Instances,PolygonMasks
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat


def from_rle_str_to_plg(encode_str):
    binary_mask = maskUtils.decode(encode_str)
    polygons = imantics.Mask(binary_mask).polygons()

    return [l.tolist() for l in polygons]

def decode_rle_mask(encode_str):
    return maskUtils.decode(encode_str)

def create_box_mask(box, img_h, img_w):
    result = np.full((img_h, img_w), False)
    x, y, h, w = box
    x, y, h, w = int(x), int(y), int(h), int(w)
    result[y:y+w, x:x+h ] = True

    return result

def get_binary_mask(polygons, height, width):
    img = Image.new('L', (width, height), 0)
    for polygon in polygons:
        formatted_polygon = []
        for i in range(0, len(polygon)-1, 2):
            formatted_polygon.append((polygon[i], polygon[i+1]))

        #print(polygon)
        #print(formatted_polygon)
        ImageDraw.Draw(img).polygon(formatted_polygon, outline=1, fill=1)

    mask = np.array(img)
    return mask

def convert_box(box):
    #import pdb;pdb.set_trace()
    x, y, h, w = box
    x1 = x
    x2 = x1 + h
    y1 = y
    y2 = y1 + w

    return [x1, y1, x2, y2]

def to_3(mask):
    img = np.zeros((mask.shape[0], mask.shape[1], 3))
    img[:, :, 0] = mask
    img[:, :, 1] = mask
    img[:, :, 2] = mask

    return img



def main():
    
    ########## all
    # d2sa
    # ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_validation_amodal.json'
    # # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_images_v1/images'
    # OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/d2sa/123/ori_gt'
    # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/d2sa/123/ori_select'

    # # cocoa
    # ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_train2014_with_classes_amodal.json'
    # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/coco2014/train2014'
    # OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/cocoa/train_ori'

    # # # kins
    # ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_amodal.json'
    # OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/kins/select_main'
    # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/kins/select_main_ori'

    ########### select
    # # kins
    # ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_amodal.json'
    # OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/kins/gt'
    # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/kins/ori'

    # # # cocoa
    # ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/cocoa/COCO_amodal_val2014_with_classes_amodal.json'
    # OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/cocoa/gt'
    # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/cocoa/ori'

    # # # d2sa
    # ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/d2s/d2s_amodal_annotations_v1/D2S_amodal_validation_amodal.json'
    # OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/d2sa/gt'
    # IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/d2sa/ori'

    # kins
    ANNOT_FILE = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/kins/test_amodal.json'
    OUTPUT_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/matrix/gt'
    IMG_DIR = '/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/matrix/ori'

    coco = COCO(ANNOT_FILE)
    cat_ids = coco.getCatIds()

    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, coco_img_key in enumerate(coco.imgs.keys()):
        img_info = coco.loadImgs(coco_img_key)[0]

        # # ori
        # try:
        #     cv_img = Image.open('{}/{}'.format(IMG_DIR,img_info['file_name']))
        #     cv_img.save('{}/{}'.format(OUTPUT_DIR,img_info['file_name']))
        #     print(img_info['file_name'])
        # except:
        #     print("the folder does not contain the image")
        #     continue

        # gt
        try:
            cv_img = read_image('{}/{}'.format(IMG_DIR,img_info['file_name']), 'RGB')
            print(img_info['file_name'])
        except:
            # print("the folder does not contain the image")
            continue

        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)

        visualizer = Visualizer(cv_img)
        H, W, C = cv_img.shape
        amodal_masks = []

        # ## single instance
        # for ann in anns:
        #     if ann["id"] == 50948:
        #     # if ann["id"] == 82275:
            
        #         segm = ann.get("segmentation", None)
        #         # segm = ann.get("i_segmentation", None)
        #         # segm = ann.get("bg_object_segmentation", None)
        #         #segm = img_info['amodal_full']
        #         if type(segm) == list:
        #             amodal_masks.append(get_binary_mask(segm, H, W))
        #         else:
        #             amodal_masks.append(decode_rle_mask(segm))

        #         # segm1 = get_binary_mask(ann.get("segmentation", None), H, W )
        #         # segm2 = get_binary_mask(ann.get("i_segmentation", None), H, W )
        #         # segm = (segm1 ^ segm2)
        #         # amodal_masks.append(segm)

        boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anns
            if obj["iscrowd"] == 0
        ]

        target = Instances((H,W))
        target.gt_boxes = Boxes(boxes)

        ## all instance
        v_segms = PolygonMasks([obj["i_segmentation"] for obj in anns])
        o_segms = PolygonMasks([obj["bg_object_segmentation"] for obj in anns])

        # for ann in anns:
        #     v_segm = ann.get("i_segmentation", None)
        #     o_segm = ann.get("bg_object_segmentation", None)
            # if type(v_segm) == list:
            #     v_masks.append(get_binary_mask(v_segm, H, W))
            #     o_segms.append(get_binary_mask(o_segm, H, W))
            # else:
            #     v_masks.append(decode_rle_mask(v_segm))
            #     o_segms.append(decode_rle_mask(o_segm))
            
        target.gt_i_masks = v_segms
        target.gt_o_masks = o_segms

        gt_bo_masks = []
        gt_vi_masks = []

        gt_i_masks_per_image = target.gt_i_masks.crop_and_resize(
            target.gt_boxes.tensor, 14
        )
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_vi_masks.append(gt_i_masks_per_image)
    
        # occluder - context
        gt_o_masks_per_image = target.gt_o_masks.crop_and_resize(
            target.gt_boxes.tensor, 14
        )
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_bo_masks.append(gt_o_masks_per_image)

        gt_bo_masks = cat(gt_bo_masks, dim=0)
        gt_vi_masks = cat(gt_vi_masks, dim=0)

        back_part_gt_1 = ~(gt_bo_masks | gt_vi_masks)
        gt_matrix = torch.stack(( gt_vi_masks, gt_bo_masks, back_part_gt_1 ),dim = 1).view(-1, 3, 196).float()
        gt_matrix_0 = torch.transpose(gt_matrix, 1, 2)
        final_matrix = torch.bmm(gt_matrix_0, gt_matrix)

        # print(final_matrix[0])
        # exit(0)

        for i in range(final_matrix.shape[0]):
            # 取出当前图像的掩码
            mask = final_matrix[i].cpu().numpy() * 255

            # height, width = mask.shape
            # blue_background = np.zeros((height, width, 3))
            # blue_background[:, :, 0] = 255  # 设置蓝色通道为255

            # result = blue_background[:, :, 0] * mask

            # 确保保存路径存在
            save_path = f"/media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/pap_output/pap_new/matrix/gt2"
            # 保存可视化结果
            save_file = os.path.join(save_path, f'mask_{i}.png')
            cv2.imwrite(save_file, mask)

            print(f"Visualization for image {i} saved at: {save_file}")

        exit(0)
        v_masks = np.array(v_masks)
        o_segms = np.array(o_segms)

        my_colors = [
                    np.array([1., 0., 0.], dtype=np.float32),np.array([0. , 0., 1.], dtype=np.float32),
                    np.array([0., 1., 1.], dtype=np.float32),np.array([0.5, 1., 0.], dtype=np.float32),
                    np.array([1., 0., 0.], dtype=np.float32), np.array([0.5, 1., 0.], dtype=np.float32),
                    np.array([0.5, 0.5, 0.], dtype=np.float32)
                ]

        visualizer.draw_masks(
                amodal_masks, name=join(OUTPUT_DIR, img_info['file_name']),
                # colors=my_colors,
            )
        print("end")


main()
