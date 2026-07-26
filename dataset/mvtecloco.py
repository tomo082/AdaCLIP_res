import os
from pathlib import Path

from .base_dataset import BaseDataset
from config import DATA_ROOT


MVTECLOCO_CLS_NAMES = [
    'breakfast_box',
    'juice_bottle',
    'pushpins',
    'screw_bag',
    'splicing_connectors',
]

MVTECLOCO_ROOT = str(Path(
    os.environ.get(
        'ADACLIP_MVTECLOCO_ROOT',
        os.path.join(DATA_ROOT, 'mvtec_loco_anomaly_detection'),
    )
).expanduser().resolve())


class MVTecLOCODataset(BaseDataset):
    """MVTec LOCO AD dataset using its official MVTec-style layout."""

    def __init__(
        self,
        transform,
        target_transform,
        clsnames=MVTECLOCO_CLS_NAMES,
        aug_rate=0.0,
        root=MVTECLOCO_ROOT,
        training=True,
    ):
        super(MVTecLOCODataset, self).__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training,
        )
