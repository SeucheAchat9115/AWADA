"""Tests for Sim10kDataset._parse_annotation and the full Sim10kDataset class."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from src.datasets.sim10k import CLASS_NAMES, Sim10kDataset


def _write_voc_xml(path, objects):
    """Write a minimal PASCAL VOC XML annotation file.

    objects: list of (name, x1, y1, x2, y2)
    """
    root = ET.Element('annotation')
    for name, x1, y1, x2, y2 in objects:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = name
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x1)
        ET.SubElement(bndbox, 'ymin').text = str(y1)
        ET.SubElement(bndbox, 'xmax').text = str(x2)
        ET.SubElement(bndbox, 'ymax').text = str(y2)
    tree = ET.ElementTree(root)
    tree.write(path)


class TestSim10kParseAnnotation:
    def test_parses_single_car(self, tmp_path):
        xml_path = str(tmp_path / 'test.xml')
        _write_voc_xml(xml_path, [('car', 10, 20, 100, 200)])
        boxes, labels = Sim10kDataset._parse_annotation(xml_path)
        assert len(boxes) == 1
        assert boxes[0] == [10.0, 20.0, 100.0, 200.0]
        assert labels[0] == 1  # 'car' is class 1

    def test_ignores_non_car_objects(self, tmp_path):
        xml_path = str(tmp_path / 'test.xml')
        _write_voc_xml(xml_path, [('person', 10, 20, 100, 200), ('car', 5, 5, 50, 60)])
        boxes, labels = Sim10kDataset._parse_annotation(xml_path)
        assert len(boxes) == 1
        assert labels[0] == 1

    def test_tiny_box_skipped(self, tmp_path):
        """Boxes with w<=5 or h<=5 should be skipped."""
        xml_path = str(tmp_path / 'test.xml')
        _write_voc_xml(xml_path, [('car', 10, 10, 13, 13)])  # 3x3 box
        boxes, labels = Sim10kDataset._parse_annotation(xml_path)
        assert len(boxes) == 0

    def test_missing_file_returns_empty(self):
        """Missing annotation file returns empty lists without raising."""
        boxes, labels = Sim10kDataset._parse_annotation('/nonexistent/path.xml')
        assert boxes == []
        assert labels == []

    def test_multiple_cars(self, tmp_path):
        xml_path = str(tmp_path / 'test.xml')
        _write_voc_xml(xml_path, [
            ('car', 10, 10, 100, 100),
            ('car', 200, 200, 400, 400),
        ])
        boxes, labels = Sim10kDataset._parse_annotation(xml_path)
        assert len(boxes) == 2

    def test_class_names_contains_car(self):
        assert 'car' in CLASS_NAMES

    def test_parse_annotation_car_label_is_one_indexed(self, tmp_path):
        xml_path = str(tmp_path / 'test.xml')
        _write_voc_xml(xml_path, [('car', 0, 0, 100, 100)])
        _, labels = Sim10kDataset._parse_annotation(xml_path)
        assert labels[0] == 1  # 1-indexed


@pytest.fixture()
def sim10k_root(tmp_path):
    """Minimal Sim10k directory structure with one image and one annotation."""
    images_dir = tmp_path / 'images'
    ann_dir = tmp_path / 'Annotations'
    images_dir.mkdir()
    ann_dir.mkdir()

    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(str(images_dir / '00001.jpg'))

    _write_voc_xml(str(ann_dir / '00001.xml'), [('car', 10, 10, 50, 50)])
    return str(tmp_path)


class TestSim10kDataset:
    def test_len(self, sim10k_root):
        ds = Sim10kDataset(sim10k_root)
        assert len(ds) == 1

    def test_getitem_returns_image_and_target(self, sim10k_root):
        ds = Sim10kDataset(sim10k_root)
        image, target = ds[0]
        assert image.shape[0] == 3  # RGB

    def test_getitem_target_keys(self, sim10k_root):
        ds = Sim10kDataset(sim10k_root)
        _, target = ds[0]
        assert 'boxes' in target
        assert 'labels' in target
        assert 'image_id' in target

    def test_getitem_box_values(self, sim10k_root):
        ds = Sim10kDataset(sim10k_root)
        _, target = ds[0]
        boxes = target['boxes']
        assert boxes.shape == (1, 4)
        assert boxes[0, 0].item() == pytest.approx(10.0)

    def test_no_annotation_file_returns_empty_boxes(self, tmp_path):
        images_dir = tmp_path / 'images'
        ann_dir = tmp_path / 'Annotations'
        images_dir.mkdir()
        ann_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(str(images_dir / '00001.jpg'))
        # No .xml file written

        ds = Sim10kDataset(str(tmp_path))
        _, target = ds[0]
        assert target['boxes'].shape == (0, 4)
