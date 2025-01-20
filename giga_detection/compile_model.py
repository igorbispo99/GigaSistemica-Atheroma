import torch
from mmdet.models import build_detector
from mmengine.config import Config
from mmengine.runner import load_checkpoint


def convert_to_standalone(config_file, checkpoint_file, output_file, device='cpu'):
    # Load the config
    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None  # Remove training-related parts

    # Build the model
    model = build_detector(cfg.model)
    model.cfg = cfg  # Attach config to the model
    model.to(device)
    model.eval()

    # Load weights
    load_checkpoint(model, checkpoint_file, map_location=device)

    # Save the standalone PyTorch model
    torch.save(model.state_dict(), output_file)
    print(f"Standalone model saved to {output_file}")


# Example usage
config_file = '/mnt/data/Code/Igor/mmdetection/configs/empirical_attention/faster-rcnn_r50-attn0010_fpn_1x_coco.py'
checkpoint_file = 'models/empirical_attention/epoch_100.pth'
output_file = 'faster_rcnn_standalone.pth'
convert_to_standalone(config_file, checkpoint_file, output_file, device='cpu')
