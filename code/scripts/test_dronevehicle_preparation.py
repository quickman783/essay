"""Regression checks for real image encoding, crop geometry and separate labels."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parent / 'DMM-main/tools'


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


images = load('data_process')
labels = load('VOC2DOTA')


def xml(name, x):
    return ('<annotation><object><name>' + name + '</name><polygon>'
            f'<x1>{x}</x1><y1>110</y1><x2>{x+20}</x2><y2>110</y2>'
            f'<x3>{x+20}</x3><y3>130</y3><x4>{x}</x4><y4>130</y4>'
            '</polygon></object></annotation>')


class PreparationTests(unittest.TestCase):
    def test_invalid_polygon_and_typo(self):
        self.assertIsNone(labels.format_line([(1, 1)] * 4, 'car', 0, -100))
        self.assertEqual(labels.normalize_class('truvk', 'dronevehicle'), 'truck')

    def test_real_png_and_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'source.jpg', Path(tmp) / 'target.png'
            image = np.random.default_rng(42).integers(0, 256, (712, 840, 3), dtype=np.uint8)
            cv2.imwrite(str(source), image)
            images.convert(str(source), str(target), images.DRONEVEHICLE_CROP)
            self.assertEqual(target.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            np.testing.assert_array_equal(cv2.imread(str(target)), cv2.imread(str(source))[100:612, 100:740])
            images.convert(str(source), str(target))
            self.assertEqual(target.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_nested_layout_and_independent_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('trainlabel', 'trainlabelr'):
                (root / 'source/train/train' / folder).mkdir(parents=True)
            (root / 'source/train/train/trainlabel/00001.xml').write_text(xml('car', 120))
            (root / 'source/train/train/trainlabelr/00001.xml').write_text(xml('truck', 140))
            labels.prepare_split(str(root / 'source'), str(root / 'output'), 'train',
                                 'dronevehicle', -100, None, False)
            vi = (root / 'output/train/annofiles_vi/00001.txt').read_text().split()
            ir = (root / 'output/train/annofiles_ir/00001.txt').read_text().split()
            self.assertEqual(vi[8], 'car')
            self.assertEqual(ir[8], 'truck')
            self.assertEqual(min(map(float, vi[:8:2])), 20)
            self.assertEqual(min(map(float, ir[:8:2])), 40)

    def test_empty_label_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ('testlabel', 'testlabelr'):
                (root / 'source/test' / folder).mkdir(parents=True)
                (root / 'source/test' / folder / '00001.xml').write_text('<annotation/>')
            labels.prepare_split(str(root / 'source'), str(root / 'output'), 'test',
                                 'dronevehicle', -100, None, False)
            self.assertEqual((root / 'output/test/annofiles_ir/00001.txt').read_text(), '')


if __name__ == '__main__':
    unittest.main()
