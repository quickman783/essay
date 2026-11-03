# VEDAI annotation class ids:
# 1: car, 2: truck, 4: tractor, 5: camping car, 9: van, 11: pickup
# Ignored VEDAI classes (6/23: boat, 7: motorcycle, 8: bus, 10: other, 12: large, 31: plane).

import argparse
import os
from collections import OrderedDict

import pandas as pd
import xml.etree.ElementTree as ET

COLUMNS = ['x_center', 'y_center', 'orientation', 'class', 'is_contained', 'is_occluded',
		   'corner1_x', 'corner2_x', 'corner3_x', 'corner4_x',
		   'corner1_y', 'corner2_y', 'corner3_y', 'corner4_y']

# VEDAI class id -> detection class name. Only these 6 classes are exported.
CLASS_NAMES = OrderedDict([
	(1, 'car'),
	(2, 'truck'),
	(4, 'tractor'),
	(5, 'camping car'),
	(9, 'van'),
	(11, 'pickup'),
])

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

COUNT_OBJECT = 0
COUNT_IMAGE = 0
COUNT_EMPTY_IMAGE = 0
COUNT_CLASS = OrderedDict((name, 0) for name in CLASS_NAMES.values())


def to_int(value):
	try:
		return int(float(value))
	except (TypeError, ValueError):
		return None


def clamp(value, lower, upper):
	return max(lower, min(value, upper))


def update_annotations(filename, annotation_dir, output_dir):
	global COUNT_OBJECT
	global COUNT_IMAGE
	global COUNT_EMPTY_IMAGE

	data = pd.read_csv(os.path.join(annotation_dir, filename), sep=r'\s+', index_col=None, header=None, names=COLUMNS)
	imagename = filename.split('.')[0]
	print('Work on: {0}'.format(imagename))

	annotation = ET.Element("annotation", verified="yes")
	ET.SubElement(annotation, "folder").text = 'images'
	ET.SubElement(annotation, "filename").text = '{0}.{1}'.format(imagename, 'jpg')
	ET.SubElement(annotation, "path").text = 'images/{0}.jpg'.format(imagename)

	source = ET.SubElement(annotation, "source")
	ET.SubElement(source, "database").text = 'Unknown'

	size = ET.SubElement(annotation, 'size')
	ET.SubElement(size, 'width').text = '{0}'.format(IMAGE_WIDTH)
	ET.SubElement(size, 'height').text = '{0}'.format(IMAGE_HEIGHT)
	ET.SubElement(size, 'depth').text = '3'

	ET.SubElement(annotation, 'segmented').text = '0'

	for row in range(len(data)):
		record = data.iloc[row]
		class_id = to_int(record['class'])
		if class_id not in CLASS_NAMES:
			continue

		xs = [to_int(record['corner1_x']), to_int(record['corner2_x']),
			  to_int(record['corner3_x']), to_int(record['corner4_x'])]
		ys = [to_int(record['corner1_y']), to_int(record['corner2_y']),
			  to_int(record['corner3_y']), to_int(record['corner4_y'])]
		if None in xs or None in ys:
			continue

		name = CLASS_NAMES[class_id]
		object = ET.SubElement(annotation, 'object')
		ET.SubElement(object, 'name').text = name
		ET.SubElement(object, 'pose').text = 'Unspecified'
		ET.SubElement(object, 'truncated').text = '0'
		ET.SubElement(object, 'difficult').text = '0'
		bndbox = ET.SubElement(object, 'bndbox')
		ET.SubElement(bndbox, 'xmin').text = '{0}'.format(clamp(min(xs), 0, IMAGE_WIDTH))
		ET.SubElement(bndbox, 'ymin').text = '{0}'.format(clamp(min(ys), 0, IMAGE_HEIGHT))
		ET.SubElement(bndbox, 'xmax').text = '{0}'.format(clamp(max(xs), 0, IMAGE_WIDTH))
		ET.SubElement(bndbox, 'ymax').text = '{0}'.format(clamp(max(ys), 0, IMAGE_HEIGHT))

		COUNT_CLASS[name] += 1
		COUNT_OBJECT += 1

	if len(annotation.findall('object')) == 0:
		COUNT_EMPTY_IMAGE += 1
		print('Skip: {0} (no object of the 6 target classes)'.format(imagename))
		return

	ET.ElementTree(annotation).write(os.path.join(output_dir, '{0}.xml'.format(imagename)))
	COUNT_IMAGE += 1


def default_output_dir(annotation_dir):
	norm = os.path.normpath(annotation_dir)
	base = os.path.basename(norm)
	# legacy repo layout: annotations512/ in the working directory -> image_512/
	legacy = {'annotations512': 'image_512', 'annotations1024': 'image_1024'}
	if os.path.dirname(norm) == '' and base.lower() in legacy:
		return legacy[base.lower()]
	suffix = ''.join(ch for ch in base if ch.isdigit())
	return os.path.join(os.path.dirname(norm), 'AnnotationsXML{0}'.format(suffix))


def find_annotation_dir(annotation_dir):
	if annotation_dir is not None:
		return annotation_dir
	# look for the default folder in the working directory, then in data/VEDAI
	for candidate in ('annotations512', os.path.join('data', 'VEDAI', 'Annotations512'),
					  os.path.join('..', 'data', 'VEDAI', 'Annotations512'),
					  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotations512')):
		if os.path.isdir(candidate):
			return candidate
	raise SystemExit('Annotation directory not found. Run from the code directory or pass --annotation_dir <dir>.')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate Pascal VOC XML files from VEDAI annotation txt files.')
	parser.add_argument('--annotation_dir', default=None, help='directory containing the VEDAI annotation txt files (default: annotations512, data/VEDAI/Annotations512)')
	parser.add_argument('--output_dir', default=None, help='directory the XML files are written to (default: image_512 if it exists, otherwise AnnotationsXML512 next to the annotations)')
	args = parser.parse_args()

	annotation_dir = find_annotation_dir(args.annotation_dir)
	output_dir = args.output_dir or default_output_dir(annotation_dir)
	print('Annotations: {0}'.format(annotation_dir))
	print('Output dir : {0}'.format(output_dir))

	os.makedirs(output_dir, exist_ok=True)
	for filename in sorted(os.listdir(annotation_dir)):
		if not filename.endswith('.txt'):
			continue
		# only per-image annotations (named after the image id) are converted;
		# files such as annotation512.txt or fold01.txt are not image annotations
		if not os.path.splitext(filename)[0].isdigit():
			print('Skip non-image annotation file: {0}'.format(filename))
			continue
		update_annotations(filename, annotation_dir, output_dir)

	print("Dataset generate XML DONE!")
	print("XML files: {0}, images without target class: {1}".format(COUNT_IMAGE, COUNT_EMPTY_IMAGE))
	for name, count in COUNT_CLASS.items():
		print("Count {0}:{1}".format(name, count))
	print("All count object: {0}".format(COUNT_OBJECT))