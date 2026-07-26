from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from dataset.mvtec_layout import build_mvtec_style_meta


CLASS_NAME = 'breakfast_box'


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 8), color='white').save(path)


def _write_mask(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('L', (8, 8), color=255).save(path)


class MVTecLOCOLayoutTest(unittest.TestCase):
    def _make_common_images(self, root):
        _write_image(root / CLASS_NAME / 'train' / 'good' / '000.png')
        _write_image(root / CLASS_NAME / 'test' / 'good' / '001.png')
        _write_image(
            root / CLASS_NAME / 'test' / 'logical_anomalies' / '002.png'
        )

    def test_official_nested_mask_layout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_common_images(root)
            nested_mask = (
                root / CLASS_NAME / 'ground_truth' /
                'logical_anomalies' / '002' / '000.png'
            )
            _write_mask(nested_mask)

            metadata = build_mvtec_style_meta(root, [CLASS_NAME])
            anomaly = metadata['test'][CLASS_NAME][1]

            self.assertEqual(anomaly['anomaly'], 1)
            self.assertEqual(
                anomaly['mask_path'],
                f'{CLASS_NAME}/ground_truth/logical_anomalies/002/000.png',
            )

    def test_flattened_mask_layout_remains_supported(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_common_images(root)
            flat_mask = (
                root / CLASS_NAME / 'ground_truth' /
                'logical_anomalies' / '002.png'
            )
            _write_mask(flat_mask)

            metadata = build_mvtec_style_meta(root, [CLASS_NAME])
            anomaly = metadata['test'][CLASS_NAME][1]

            self.assertEqual(
                anomaly['mask_path'],
                f'{CLASS_NAME}/ground_truth/logical_anomalies/002.png',
            )


if __name__ == '__main__':
    unittest.main()
