from random import random, randint
from typing import List, Tuple

import keras
from imgaug.augmenters import Augmenter
import cv2
import numpy as np

from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator
from neural_nets.retina_net.keras_retinanet.utils.anchors import anchor_targets_bbox, guess_shapes
from neural_nets.retina_net.keras_retinanet.utils.image import TransformParameters, preprocess_image

from common.adapters.datasets.interfaces import AbstractDataset
from common.utils.images import draw_box


class RetinaDataset(AbstractDataset, Generator):

    def __init__(self,
                 dataset_path: str,
                 simplify_classes: bool = False,
                 batch_size: int = 1,
                 max_image_side_length: int = None,
                 augmentation: Augmenter = None,
                 center_color_to_imagenet: bool = False,
                 image_scale_mode: str = 'just',
                 pre_image_scale=0.5):

        super(RetinaDataset, self).__init__(dataset_path, simplify_classes, batch_size, max_image_side_length, augmentation, center_color_to_imagenet,
                                            image_scale_mode, pre_image_scale)

        self.center_color_to_imagenet = True
        self.image_scale_mode = 'just'

        if max_image_side_length is None:
            self.max_image_side_length = 1333
            self.min_image_side_length = 800

    # ==================== BaseDataset Methods =========================
    def combine_x_y(self, x_y_list: List[Tuple], num_items: int):

        images = [None] * num_items

        all_images = None
        all_annotations = None

        for x_y, batch_idx in x_y_list:
            image_batch, target_batch = x_y
            imgs, annos = target_batch[0], target_batch[1]

            if all_images is None:
                all_images = imgs
            else:
                all_images += imgs

            if all_annotations is None:
                all_annotations = annos
            else:
                all_annotations += annos

            for image, idx in zip(image_batch, batch_idx):
                images[idx] = image

        return self.compute_inputs(images), self.compute_targets(all_images, all_annotations)

    def compile_dataset(self):
        self.group_method = 'ratio'
        self.shuffle_groups = False
        self.visual_effect_generator = None
        self.transform_generator = None
        self.image_min_side = 512
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
    def image_path(self, image_index):
        pass

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

    def get_x_y(self, indices: List[int], raw=False):
        """
        Create arrays for input and targets for Retina Net

        :param indices: List of image indices to produce
        :type indices: List[int]
        :return: Tuple of:
        - Batch of images
        - Tuple of:
        -- Batch of box sets, one set of boxes for each image
        -- Batch of label sets, one set of labels for each image
        """

        annotations = []
        batch_of_input_images, batch_of_mask_sets, batch_of_bbox_sets, batch_of_label_sets, num_labels = super(RetinaDataset, self)._get_x_y(
            indices=indices,
            autoscale=True,
            use_masks=False,
            do_preprocessing=True,
            downscale=True
        )

        # Extract boxes
        for batch, sets in enumerate(zip(batch_of_input_images, batch_of_bbox_sets, batch_of_label_sets)):
            image, box_set, label_set = sets
            print(image.shape)
            annotations.append({
                'bboxes': box_set,
                'labels': label_set
            })

            # Uncomment for DEBUG
            # ==========================
            # ==========================
            draw = image.copy()

            draw[..., 0] += 123.68  # R
            draw[..., 1] += 116.779  # G
            draw[..., 2] += 103.939  # B

            for ann in annotations:

                for box in ann.get('bboxes'):
                    draw_box(draw, [int(box[1]), int(box[0]), int(box[3]), int(box[2])], color=(255, 200, 0))
                    caption = "{} {:.3f}".format('hur', 0)

                    # print(self.labels.index(obj['name'])  )

                    cv2.putText(
                        img=draw,
                        text=caption,
                        org=(int(box[0]), int(box[1]) - 10),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1,
                        color=(255, 200, 0),
                        thickness=1)

            from matplotlib import pyplot as plt
            fig = plt.figure(figsize=(10,15))
            plt.axis('off')
            try:
                plt.imshow(draw.astype(np.uint8))
            except:
                pass
            plt.show()
            with open('train_images/{}.png'.format(randint(0, 1000)), 'wb') as f:
                fig.savefig(f, format='png')

            exit(0)
            # ==========================
            # ==========================

        # Compute regression targets
        targets = (batch_of_input_images, annotations) if raw else self.compute_targets(batch_of_input_images, annotations)
        # batch_of_input_images = self.compute_inputs(batch_of_input_images)
        return batch_of_input_images, list(targets)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        return self.get_x_y(group)
