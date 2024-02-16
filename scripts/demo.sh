# python3 setup.py build develop #--no-deps
export PYTHONPATH=$PYTHONPATH:`pwd`
#export CUDA_LAUNCH_BLOCKING=1 # for debug
export CUDA_VISIBLE_DEVICES=1

# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet_r101.yaml"
# config_path="configs/KINS-Base-RCNN-FPN-Fast-BCNet-regnet.yaml"
# config_path="configs/COCOA-Base-RCNN-FPN-Fast-BCNet.yaml"
config_path="configs/D2SA-Base-RCNN-FPN-Fast-BCNet.yaml"

weight_path="weights/d2sa.pth"

python3 demo/demo.py --config-file ${config_path} \
  --input /media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/mas_github  \
  --output /media/jiao/39b48156-5afd-4cd7-bddc-f6ecf4631a79/zhanjiao/dataset/output/mas_github_output  \
  --confidence-threshold 0.7 \
  --opts MODEL.WEIGHTS ${weight_path}
