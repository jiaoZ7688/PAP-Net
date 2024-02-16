#python3 setup.py build develop #--no-deps # for building d2
export PYTHONPATH=$PYTHONPATH:`pwd`
# export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=0

ID="kins_pap_sa"

# TODO: specify experiment config
## kins
# config_path="configs/KINS-Base-RCNN-FPN-CAM-BCNet.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet-regnet.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet-regnet_CLS_AGNOSTIC.yaml"
config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet_CLS_AGNOSTIC.yaml"
# config_path="configs/D2SA-Base-RCNN-FPN-Fast-BCNet.yaml"
# config_path="configs/COCOA-Base-RCNN-FPN-Fast-BCNet.yaml"


python3 tools/train_net.py --num-gpus 1 \
	--config-file ${config_path} 2>&1 | tee log/train_log_${ID}.txt
