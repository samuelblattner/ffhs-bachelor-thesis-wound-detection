from typing import List

from PIL import Image, ImageDraw
from imgaug.augmenters import Augmenter
import numpy as np
from utils.utils import normalize

from common.adapters.datasets.interfaces import AbstractDataset
from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator
from neural_nets.retina_net.keras_retinanet.utils.anchors import anchor_targets_bbox, guess_shapes
from neural_nets.retina_net.keras_retinanet.utils.image import TransformParameters, preprocess_image
from neural_nets.yolo_3.generator import BatchGenerator
from utils.bbox import BoundBox


class Yolo_3Dataset(AbstractDataset, BatchGenerator, Generator):
    anchors = [4, 3, 7, 10, 10, 4, 15, 7, 16, 14, 19, 3, 26, 9, 30, 17, 119, 113]

    generator = None

    def __init__(self, dataset_path: str, simplify_classes: bool = False, batch_size: int = 1, max_image_side_length: int = 512,
                 augmentation: Augmenter = None, center_color_to_imagenet: bool = False, image_scale_mode: str = 'just', pre_image_scale=0.5):

        super(Yolo_3Dataset, self).__init__(dataset_path, simplify_classes, batch_size, max_image_side_length, augmentation, False,
                                            'squash', pre_image_scale)

        self.anchors = [BoundBox(0, 0, self.anchors[2 * i], self.anchors[2 * i + 1]) for i in range(len(self.anchors) // 2)]

        self.get_item = BatchGenerator.__getitem__.__get__(self, Yolo_3Dataset)

        self.instances = self.get_instances()
        self.labels = ('Sharp Force', 'Blunt Force')
        self.downsample = 32
        self.max_box_per_image = 30
        self.min_net_size = max_image_side_length
        self.max_net_size = max_image_side_length
        self.shuffle = False
        self.jitter = 0.0
        self.norm = normalize
        self.net_h = max_image_side_length
        self.net_w = max_image_side_length
        # self.image_scale_mode = 'just'

    def _aug_image(self, instance, net_h, net_w):
        batch_of_input_images, batch_of_target_masks, batch_of_target_bboxes, batch_of_target_labels, num_classes = super(Yolo_3Dataset, self)._get_x_y(
            [instance.get('idx')], True, False, False)
        boxes = []
        for box, label in zip(batch_of_target_bboxes[0], batch_of_target_labels[0]):
            if abs(int(box[2]) - int(box[0])) <= 0 or abs(int(box[3]) - int(box[1])) <= 0:
                continue
            boxes.append({
                'name': self.labels[label],
                'xmin': int(box[0]),
                'ymin': int(box[1]),
                'xmax': int(box[2]),
                'ymax': int(box[3]),
            })

        return batch_of_input_images[0], boxes

    def get_instances(self):
        instances = []
        for i, image in enumerate(self.get_image_info()):
            mask_data = self._masks.get(image.get('id')).get('masks_raw')

            instances.append({
                'filename': image.get('path'),
                'idx': i,
                'width': image.get('width'),
                'height': image.get('height'),
                'object': [
                    {
                        'name': self.SIMPLE_CLASS_NAMES[minfo.get('category_id')],
                        'xmin': minfo.get('bbox')[1],
                        'ymin': minfo.get('bbox')[0],
                        'xmax': minfo.get('bbox')[1] + minfo.get('bbox')[3],
                        'ymax': minfo.get('bbox')[0] + minfo.get('bbox')[2]
                    } for minfo in mask_data
                ]
            })

            # ===============
            # im = Image.open(image.get('path'))
            # draw = ImageDraw.Draw(im)
            # for obj in instances[-1]['object']:
            #     draw.rectangle(
            #         [(obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax'])], None, (255,64,0), 2
            #     )
            #
            # from matplotlib import pyplot as plt
            # plt.imshow(draw)
            # exit(0)
            # ==============
        return instances

    # ==================== BaseDataset Methods =========================
    def compile_dataset(self):
        self.group_method = 'ratio'
        self.shuffle_groups = False
        self.visual_effect_generator = None
        self.transform_generator = None
        self.image_min_side = self.max_image_side_length
        self.image_max_side = self.max_image_side_length
        self.transform_parameters = TransformParameters()
        self.compute_anchor_targets = anchor_targets_bbox
        self.compute_shapes = guess_shapes
        self.preprocess_image = preprocess_image
        self.config = None

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()

    def register_image(self, group_name: str, image_id: int, path: str, width: int, height: int):
        pass

    def register_label(self, group_name: str, label_id: int, label_name: str):
        pass

    # ================== Generator Methods =============================

    def size(self):
        return len(self.get_image_info())

    def num_classes(self):
        return len(self._labels)

    def has_label(self, label):
        try:
            return self._labels.index(label) >= 0
        except ValueError:
            return False

    def has_name(self, name):
        try:
            return self._label_names.index(name) >= 0
        except ValueError:
            return False

    def name_to_label(self, name):
        if not self.has_name(name):
            return -1
        return self._labels[self._label_names.index(name)]

    def label_to_name(self, label):
        if not self.has_label(label):
            return None
        return self._label_names[self._labels.index(label)]

    def image_aspect_ratio(self, image_index):
        return self._images.get(self._img_idx_to_id(image_index)).get('width') / self._images.get(self._img_idx_to_id(image_index)).get('height')

    def load_image(self, image_index):
        img = self._load_image(self._img_idx_to_id(image_index))
        return np.asarray(img)

    def load_annotations(self, image_index):
        """
        Load all annotations for a given image at index ``image_index``.
        COCO-Dataset stores x,y,w,h, so we need to calculate x2 and y2 by addition.
        :param image_index:
        :return:
        """
        bboxes = [[b.get('bbox')[0], b.get('bbox')[1], b.get('bbox')[0] + b.get('bbox')[2], b.get('bbox')[1] + b.get('bbox')[3]] for b in
                  self._masks[self._img_idx_to_id(image_index)]['masks_raw']]
        labels = [b.get('category_id') for b in self._masks[self._img_idx_to_id(image_index)]['masks_raw']]

        return {
            'bboxes': np.asarray(bboxes) * self.IMAGE_FACTOR,
            'labels': np.asarray(labels)
        }

    def get_x_y(self, indices: List[int]):
        """
        Return an image an its corresponding ground truth boxes
        :param indices: List of indices to return from dataset
        :return: Tuple of images, boxes an zero array
        """

        return self.get_item(indices[0])

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        return self.get_x_y(group)
