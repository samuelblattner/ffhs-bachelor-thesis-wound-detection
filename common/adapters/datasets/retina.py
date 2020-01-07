from typing import List

import cv2
import numpy as np
from imgaug.augmenters import Augmenter

from common.utils.images import draw_box
from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator
from neural_nets.retina_net.keras_retinanet.utils.anchors import anchor_targets_bbox, guess_shapes
from neural_nets.retina_net.keras_retinanet.utils.image import TransformParameters, preprocess_image

from common.adapters.datasets.interfaces import AbstractDataset


class RetinaDataset(AbstractDataset, Generator):

    def __init__(self, dataset_path: str, simplify_classes: bool = False, batch_size: int = 1, max_image_side_length: int = 512,
                 augmentation: Augmenter = None, center_color_to_imagenet: bool = False, image_scale_mode: str = 'square', pre_image_scale=0.5):

        super(RetinaDataset, self).__init__(dataset_path, simplify_classes, batch_size, max_image_side_length, augmentation, center_color_to_imagenet,
                                            image_scale_mode, pre_image_scale)

        self.center_color_to_imagenet = True
    # ==================== BaseDataset Methods =========================
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
        batch_of_input_images, batch_of_mask_sets, batch_of_bbox_sets, batch_of_label_sets, num_labels = super(RetinaDataset, self)._get_x_y(indices, autoscale=True, use_masks=False, do_preprocessing=True)
        batch_size = len(batch_of_input_images)

        max_mask_set_size = 0
        for mask_set in batch_of_bbox_sets:
            max_mask_set_size = max(max_mask_set_size, mask_set.shape[0])

        # Prepare arrays
        bboxes = np.zeros((batch_size, max_mask_set_size, 4))
        labels = np.zeros((batch_size, max_mask_set_size, num_labels + 1))

        # Reset all to 'ignore' (-1)
        bboxes[:, :, -1] = -1
        labels[:, :, -1] = -1

        annotations = []

        for batch, sets in enumerate(zip(batch_of_bbox_sets, batch_of_label_sets)):

            mask_set, label_set = sets
            bboxes[batch, :mask_set.shape[0]] = mask_set
            annotations.append({
                'bboxes': bboxes[batch],
                'labels': label_set
            })

            # for b, box in enumerate(bboxes[batch]):
            #     y1, x1, y2, x2 = box
            #     bboxes[batch][b] = np.array((x1, y1, x2, y2))

        # ==========================
        # ==========================
        # draw = batch_of_input_images[0].copy()
        #
        # draw[..., 0] += 123.68  # R
        # draw[..., 1] += 116.779  # G
        # draw[..., 2] += 103.939  # B
        #
        # print(draw.shape)
        # for ann in annotations:
        #     for box in ann.get('bboxes'):
        #         draw_box(draw, [int(box[1]), int(box[0]), int(box[3]), int(box[2])], color=(255, 200, 0))
        #         caption = "{} {:.3f}".format('hur', 0)
        #
        #         # print(self.labels.index(obj['name'])  )
        #
        #         cv2.putText(
        #             img=draw,
        #             text=caption,
        #             org=(int(box[0]), int(box[1]) - 10),
        #             fontFace=cv2.FONT_HERSHEY_PLAIN,
        #             fontScale=1,
        #             color=(255, 200, 0),
        #             thickness=1)
        #
        # # Image.fromarray(draw.astype(np.uint8)).show()
        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(20, 20))
        # plt.axis('off')
        # try:
        #     plt.imshow(draw.astype(np.uint8))
        # except:
        #     pass
        # plt.show()
        #
        # # print(batch_of_input_images.shape)
        #

        # exit(0)
        # ==========================
        # ==========================

        targets = self.compute_targets(batch_of_input_images, annotations)

        batch_of_input_images = np.asarray(batch_of_input_images, dtype=np.float32)
        return batch_of_input_images, list(targets)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        return self.get_x_y(group)
