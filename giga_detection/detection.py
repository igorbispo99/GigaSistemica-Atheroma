import mmcv

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

config_file = r'/mnt/data/Code/Igor/mmdetection/configs/empirical_attention/faster-rcnn_r50-attn0010_fpn_1x_coco.py'
checkpoint_file = r'models/empirical_attention/epoch_100.pth'

img = 't.jpg'

device = 'cpu'  # ou 'cpu' se você não tiver uma GPU disponível
model = init_detector(config_file, checkpoint_file, device=device)
result = inference_detector(model, img)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')

visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    show=True,
    out_file='ATCLIN2016-1.jpg'
)


