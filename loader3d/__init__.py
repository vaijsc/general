
from loader3d.scannet_loader import ScanNetReader

__all__ = ["ScanNetReader", "build_dataset"]


def build_dataset(root_path, cfg):
    if cfg.data.dataset_name == "scannet200" or cfg.data.dataset_name == "scannetpp" or cfg.data.dataset_name == 'scannetpp_benchmark':
        return ScanNetReader(root_path, cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

