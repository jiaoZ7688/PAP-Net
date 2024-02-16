from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg


def add_config(cfg):
    # AISFormer config
    cfg.MODEL.AISFormer = CN()
    cfg.MODEL.ROI_MASK_HEAD.CUSTOM_NAME = 'AISFormer'
    cfg.MODEL.ROI_MASK_HEAD.VERSION = 0  # work on eval stage
    cfg.MODEL.AISFormer.USE = True
    cfg.MODEL.AISFormer.JUSTIFY_LOSS = True # if true, use invisible mask
    cfg.MODEL.AISFormer.MASK_NUM = 4
    cfg.MODEL.AISFormer.MASK_LOSS_ONLY = False

    # transformer layers 
    cfg.MODEL.AISFormer.N_LAYERS = 1
    cfg.MODEL.AISFormer.N_HEADS = 2

    # boundary loss
    cfg.MODEL.AISFormer.BO_LOSS = False

    # amodal eval
    cfg.MODEL.AISFormer.AMODAL_EVAL = True

    # extra
    cfg.MODEL.ALL_LAYERS_ROI_POOLING = False
    cfg.MODEL.IS_USE_MSDETR = False # use deformable detr in fpn
    cfg.MODEL.Learned_Query_LOSS = False # use learned query
    cfg.DICE_LOSS = False
    # cfg.SOLVER.OPTIMIZER = "AdamW"
    cfg.SOLVER.OPTIMIkZER = "SGD"

    # FCOS########################
    cfg.MODEL.FCOS = CN()
    cfg.MODEL.FCOS.NUM_CLASSES = 7
    cfg.MODEL.FCOS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
    cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.FCOS.IOU_LOSS_TYPE = 'giou'
    cfg.MODEL.FCOS.BOX_QUALITY = 'iou'
    cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 1.5
    cfg.MODEL.FCOS.NORM_REG_TARGETS = True
    cfg.MODEL.FCOS.CENTERNESS_ON_REG = True
    cfg.MODEL.FCOS.INFERENCE_TH = 0.03
    cfg.MODEL.FCOS.NMS_TH = 0.6
    cfg.MODEL.FCOS.PRE_NMS_TOP_N = 1000
    cfg.MODEL.FCOS.USE_DCN_IN_TOWER = False
    cfg.MODEL.FCOS.OBJECT_SIZE_OF_INTEREST = [
        [-1, 64],
        [64, 128],
        [128, 256],
        [256, 512],
        [512, float("inf")],
    ]
    cfg.MODEL.FCOS.TRAIN_PART = 'all'       
    cfg.MODEL.FCOS.SCORE_THRESH_TEST = 0.03
    cfg.MODEL.FCOS.TOPK_CANDIDATES_TEST = 1000
    cfg.MODEL.FCOS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.FCOS.NUM_CONVS = 4
    cfg.MODEL.FCOS.PRIOR_PROB = 0.01
    cfg.MODEL.FCOS.NORM = 'GN'

    # ###########################

if __name__== '__main__':
    cfg = get_cfg()
    add_config(cfg)
