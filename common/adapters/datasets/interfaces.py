import json
import re
import sys
from abc import ABCMeta, abstractmethod
from os.path import join
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from imgaug import BoundingBoxesOnImage, BoundingBox
from imgaug.augmenters import Augmenter, CropToFixedSize

from common.utils.images import resize_image, resize_mask, extract_bboxes


class AbstractDataset:
    """
    Generator to produce input, target pairs to train neural nets
    on full body shot wound images
    """

    __metaclass__ = ABCMeta

    # Statics
    # =======

    BG_LAST = False

    #: Factor to multiply size of input image by in order to reduce memory load
    IMAGE_FACTOR = 0.5

    #: Clusters to use when simplifying wound classes into two groups
    CLASS_CLUSTERS = {
        0: (
            1, 2, 3, 4, 5, 14
        ),
        1: (
            6, 7, 8, 9, 10, 11, 12, 13, 15
        )
    }

    #: Names for the clusters
    SIMPLE_CLASS_NAMES = {
        0: 'Sharp Force',
        1: 'Blunt Force'
    }

    # Properties
    # ==========
    _num_images: int = 0
    _num_classes: int = 0
    _labels: List[int] = []
    _label_names: List[str] = []

    #: Dictionary to hold ground truth images by image id
    _images = {}
    __image_info = []

    #: Translation dictionary to translate from 0-based image index to actual image id from externally loaded dataset
    __image_id_map = {}

    #: Dictionary to hold ground truth masks for a given ground truth image id
    _masks = {}
    _annotations = {}

    #: Current image index for iterator
    __cur_img_idx: int = 0

    # Methods
    # =======

    def __init__(self, dataset_path: str, simplify_classes: bool = False, batch_size: int = 1, max_image_side_length: int = 512,
                 augmentation: Augmenter = None, center_color_to_imagenet: bool = False, image_scale_mode: str = 'square', pre_image_scale=0.5):
        """
        Initialization of dataset generator.

        :param dataset_path: Directory path to dataset
        :type dataset_path: str

        :param simplify_classes: Whether to cluster wound classes into two basic classes or to use all classes individually
        :type simplify_classes: bool

        :param batch_size: Number of input/target pairs delivered in one batch
        :type batch_size: int

        :param max_image_side_length: Max length to be used for the longer image side
        :type max_image_side_length: int

        :param augmentation: imgaug augmentation to apply to every image
        :type augmentation: Augmenter

        :param center_color_to_imagenet: Whether to center channel data
        :type center_color_to_imagenet: bool

        """
        self.IMAGE_FACTOR = pre_image_scale
        self._masks = {}
        self._images = {}
        self.__image_id_map = {}
        self.__image_info = []
        self.__cur_img_idx: int = 0
        self._labels = []
        self._label_names = []

        self.simplify_classes = simplify_classes
        self.batch_size = batch_size
        self.max_image_side_length = max_image_side_length
        self.augmentation = augmentation
        self.center_color_to_imagenet = center_color_to_imagenet
        self.image_scale_mode = image_scale_mode

        self.__load_dataset(dataset_path)

    @classmethod
    def create_datasets(
            cls,
            train_dataset_path: str, val_dataset_path: str = None, test_dataset_path: str = None,
            dataset_split: Tuple = None, shuffle: bool = False, shuffle_seed: int = None, split_by_filename_base: bool = False,
            max_examples_per_filename_base=0, **kwargs):
        """
        Factory method to create training-, validation- and test data from an external dataset.
        If no path is indicated for validation and test dataset, and a dataset_split tuple is
        provided, the train dataset will be split up into training, validation and test data accordingly.

        :param train_dataset_path: Path of directory to training dataset
        :type train_dataset_path: str

        :param val_dataset_path: Path of directory to validation dataset (optional)
        :type val_dataset_path: str

        :param test_dataset_path: Path of directory to test dataset (optional)
        :type test_dataset_path: str

        :param dataset_split: Tuple containing ratios of how the training dataset should be split up into training, validation and test data.
        Ratios must add up to 1.0
        :type dataset_split: Tuple[float, float, float]

        :param shuffle: Defines if the datasets should be shuffled. If False, data will be read as is from external dataset file
        :type shuffle: bool

        :param shuffle_seed: Any int number to define np.random seed in order to make random choice deterministic
        :type shuffle_seed: int

        :param kwargs: Further kwargs to pass along to instances (see __init__)

        :return: A tuple containing training, validation and test dataset
        :rtype Tuple[Dataset, Dataset, Dataset]
        """

        # Create dataset instances
        train_dataset = cls(dataset_path=train_dataset_path, **kwargs)

        kwargs.pop('augmentation')
        val_dataset = cls(val_dataset_path or train_dataset_path, **kwargs)
        test_dataset = cls(test_dataset_path or train_dataset_path, **kwargs)

        # Split train dataset if no separate paths for validation and training datasets were specified
        if val_dataset_path is None and test_dataset_path is None and dataset_split:

            all_image_infos = train_dataset.get_image_info()

            train_ratio, val_ratio, test_ratio = dataset_split
            n_images = len(all_image_infos)
            n_train = int(train_ratio * n_images)
            n_val = int(val_ratio * n_images)

            if shuffle:
                if shuffle_seed is not None:
                    np.random.seed(shuffle_seed)

                train_image_ids = list(np.random.choice(
                    a=range(n_images),
                    size=n_train,
                    replace=False,
                ))

                remaining_image_ids = [image_id for image_id in range(n_images) if image_id not in train_image_ids]

                val_image_ids = list(np.random.choice(
                    a=remaining_image_ids,
                    size=n_val,
                    replace=False
                ))
                test_image_ids = []
            else:
                train_image_ids = [i for i in range(n_train)]
                val_image_ids = [i for i in range(n_train, n_train + n_val)]
                test_image_ids = []

            train_image_infos = []
            val_image_infos = []
            test_image_infos = []

            test_image_ids = list(filter(lambda i: i not in train_image_ids and i not in val_image_ids, range(n_images)))

            if split_by_filename_base:
                checked_ids = []
                while len(checked_ids) < len(all_image_infos):
                    for image_id in range(n_images):

                        checked_ids.append(image_id)

                        image_info = all_image_infos[image_id]
                        file_name_base = re.findall(r'([\w\d]{32}-)', image_info.get('path'))

                        if not file_name_base:
                            continue

                        file_name_base = file_name_base[0]

                        train_deficit = n_train - len(train_image_ids)
                        val_deficit = n_val - len(val_image_ids)
                        test_deficit = n_images - n_train - n_val - len(test_image_ids)

                        if train_deficit > val_deficit and train_deficit > test_deficit:
                            target = train_image_ids
                            sources = (val_image_ids, test_image_ids)
                        elif val_deficit > train_deficit and val_deficit > test_deficit:
                            target = val_image_ids
                            sources = (test_image_ids, train_image_ids)
                        else:
                            target = test_image_ids
                            sources = (train_image_ids, val_image_ids)

                        for source in sources:

                            remove_ids = []
                            for image_id in source:
                                if file_name_base in all_image_infos[image_id].get('path'):
                                    target.append(image_id)
                                    remove_ids.append(image_id)

                            for remove_id in remove_ids:

                                source.pop(source.index(remove_id))

            if max_examples_per_filename_base > 0:
                file_name_bases_count = {}
                images_info_sets = (
                    (train_image_ids, train_image_infos),
                    (val_image_ids, val_image_infos),
                    (test_image_ids, test_image_ids)
                )
                for image_ids, image_infos in images_info_sets:
                    remove_ids = []
                    for image_id, image_info in zip(image_ids, image_infos):

                        file_name_base = re.findall(r'([\w\d]{32}-)', image_info.get('path'))
                        if not file_name_base:
                            continue

                        file_name_base = file_name_base[0]

                        if file_name_bases_count.get(file_name_base, 0) >= max_examples_per_filename_base:
                            remove_ids.append(image_id)
                            continue

                        file_name_bases_count.setdefault(file_name_base, 0)
                        file_name_bases_count[file_name_base] += 1

                    for remove_id in remove_ids:
                        image_infos.pop(image_ids.index(remove_id))
                        image_ids.remove(image_ids.index(remove_id))

            for image_id in range(n_images):
                if image_id in train_image_ids:
                    train_image_infos.append(all_image_infos[image_id])
                elif image_id in val_image_ids:
                    val_image_infos.append(all_image_infos[image_id])
                else:
                    test_image_infos.append(all_image_infos[image_id])
                    test_image_ids.append(image_id)

            # print('Train:')
            # for info in train_image_infos:
            #     print(info.get('path'), end='')
            #     for other in (val_image_infos, test_image_infos):
            #         for info2 in other:
            #             if info.get('path') == info2.get('path'):
            #                 print('FAIL')
            #                 raise ValueError()
            #     else:
            #         print('')
            #
            # print('Val:')
            # for info in val_image_infos:
            #     print(info.get('path'), end = '')
            #     for info2 in test_image_infos:
            #         if info.get('path') == info2.get('path'):
            #             print('FAIL')
            #             raise ValueError()
            #     else:
            #         print('')
            # print('Test:')
            # for info in test_image_infos:
            #     print(info.get('path'))

            train_dataset.set_image_info(train_image_infos)
            val_dataset.set_image_info(val_image_infos)
            test_dataset.set_image_info(test_image_infos)

            train_dataset.generate_image_id_map()
            val_dataset.generate_image_id_map()
            test_dataset.generate_image_id_map()

        # Prepare datasets
        train_dataset.compile_dataset()
        val_dataset.compile_dataset()
        test_dataset.compile_dataset()

        return train_dataset, val_dataset, test_dataset

    def __extract_image_infos_from_annotations_file(self, dataset_path, content):

        infos_dict = json.loads(content)

        for i, image_info in enumerate(infos_dict.get('images', [])):
            image_info['width'] = int(image_info['width'] * self.IMAGE_FACTOR)
            image_info['height'] = int(image_info['height'] * self.IMAGE_FACTOR)

            image_info['path'] = join(dataset_path, image_info.get('path'))
            self._images[image_info.get('id')] = image_info
            self.__image_id_map[i] = image_info.get('id')
            self.__image_info.append(image_info)

            self.register_image(
                group_name='puppet',
                image_id=image_info.get('id'),
                path=image_info.get('path'),
                width=image_info.get('width'),
                height=image_info.get('height')
            )

    def __extract_mask_infos_from_annotations_file(self, content):
        infos_dict = json.loads(content)
        for mask_info in infos_dict.get('annotations', []):
            if self.simplify_classes:
                mask_info.update({
                    'category_id': self.get_simplified_label(mask_info.get('category_id'))
                })
            self._masks.setdefault(mask_info.get('image_id'), {}).setdefault('masks_raw', []).append(mask_info)

    def __extract_category_infos_from_annotations_file(self, content):
        infos_dict = json.loads(content)

        labels_to_add = {}

        # if not self.BG_LAST:
        #     labels_to_add.update({
        #         0: 'Background'
        #     })
        #
        if self.simplify_classes:
            labels_to_add.update({
                0: 'Sharp Force',
                1: 'Blunt Force'
            })
        else:
            labels_to_add.update({
                cat.get('id'): cat.get('Name', 'Unnamed label') for cat in infos_dict.get('categories', [])
            })

        for label, name in labels_to_add.items():
            if label not in self._labels and name not in self._label_names:
                self._labels.append(label)
                self._label_names.append(name)
                self.register_label(
                    'puppet',
                    label_id=label,
                    label_name=name
                )

        # if self.BG_LAST:
        self._labels.append(len(self._labels))
        self._label_names.append('bg')
        self.register_label(
            'puppet',
            label_id=len(self._labels),
            label_name='bg'
        )

    def __load_dataset(self, dataset_path: str = None):

        with open(join(dataset_path, 'annotations.json'), 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            self.__extract_image_infos_from_annotations_file(dataset_path, content)
            self.__extract_mask_infos_from_annotations_file(content)
            self.__extract_category_infos_from_annotations_file(content)

    def __len__(self):
        return len(self.__image_info)

    def __iter__(self):
        self.__cur_img_idx = 0
        return self

    def _get_x_y(self, indices: List[int], autoscale: bool = True, use_masks: bool = True, do_preprocessing: bool = False):

        batch_of_input_images = []
        batch_of_target_masks = []
        batch_of_target_labels = []

        # Iterate over num batches
        for batch_item in indices:

            # Load image, masks and labels
            image = self.load_image(batch_item)
            masks, labels = self.load_mask(batch_item, as_box=not use_masks)

            initial_shape = image.shape
            initial_width = image.shape[1]
            has_crop = False

            # Apply augmentations if specified
            if self.augmentation:
                import imgaug

                MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                                   "Fliplr", "Flipud", "CropAndPad",
                                   "Affine", "PiecewiseAffine"]

                def hook(images, augmenter, parents, default):
                    """Determines which augmenters to apply to masks."""
                    return augmenter.__class__.__name__ in MASK_AUGMENTERS

                # Store shapes before augmentation to compare
                image_shape = image.shape
                mask_shape = masks.shape

                # Make augmenters deterministic to apply similarly to images and masks
                for child_augmenter in self.augmentation.get_all_children():
                    if isinstance(child_augmenter, CropToFixedSize):
                        has_crop = True
                        min_side = min(*image.shape[:2])
                        child_augmenter.size = (min_side, min_side)

                det = self.augmentation.to_deterministic()
                image = det.augment_image(image.astype(np.uint8))
                # Change mask to np.uint8 because imgaug doesn't support np.bool

                # for mask in masks:
                #     print([(m[0], m[1], m[2], m[3]) for m in masks])
                if use_masks:
                    masks = det.augment_image(masks.astype(np.uint8), hooks=imgaug.HooksImages(activator=hook))
                else:
                    bbs = BoundingBoxesOnImage([
                        BoundingBox(x1=m[0], x2=m[2], y1=m[1], y2=m[3]) for m in masks
                    ], shape=initial_shape)

                    aug_boxes = []
                    w, h = image.shape[1], image.shape[0]

                    for b in det.augment_bounding_boxes(bbs).bounding_boxes:
                        aug_boxes.append(
                            [
                                max(0, min(w, b.x1)),
                                max(0, min(h, b.y1)),
                                max(0, min(w, b.x2)),
                                max(0, min(h, b.y2)),
                            ]
                        )
                    masks = np.array(aug_boxes)

                # Verify that shapes didn't change
                # assert image.shape == image_shape, "Augmentation shouldn't change image size"
                # assert masks.shape == mask_shape, "Augmentation shouldn't change mask size"
                # Change mask back to bool
                # ret_mask = ret_mask.astype(np.bool)

            # If max image side length was specified, resize image and masks
            if self.max_image_side_length is not None and autoscale:
                old_height, old_width = image.shape[0], image.shape[1]
                ratio = old_height / old_width

                if self.image_scale_mode == 'squash':
                    image = cv2.resize(image, (self.max_image_side_length, self.max_image_side_length))
                    s_w = image.shape[1] / old_width
                    s_h = image.shape[0] / old_height

                    if use_masks:
                        raise RuntimeError('Squashin masks is not implemented')
                    else:
                        masks[:, 0] = np.multiply(masks[:, 0], s_w)
                        masks[:, 2] = np.multiply(masks[:, 2], s_w)
                        masks[:, 1] = np.multiply(masks[:, 1], s_h)
                        masks[:, 3] = np.multiply(masks[:, 3], s_h)

                image, w, scale, p, c = resize_image(
                    image, max_dim=self.max_image_side_length, min_dim=self.max_image_side_length
                )

                # print(w, p, c, scale)

                if use_masks:
                    masks = resize_mask(
                        masks, scale=scale, padding=p
                    )
                else:
                    # masks2 = []
                    # print(masks)
                    masks = np.multiply(masks, scale)

                    if p[0][0] > 0:
                        masks[:, 1] += p[0][0]
                        masks[:, 3] += p[0][0]

                    if has_crop:
                        masks[:, 0] += p[1][0]
                        masks[:, 2] += p[1][0]
                        masks[:, 1] += p[0][0]
                        masks[:, 3] += p[0][0]

                #     for mask in masks:
                #         masks2.append(
                #             [mask[0] - w[1], mask[1] - w[0], mask[2] - w[1], mask[3] - w[0]]
                #         )
                #
                #     masks = np.array(masks2)
            if autoscale and self.image_scale_mode == 'just':
                new_width = int((initial_width * scale))
                remove = int((image.shape[1] - new_width) / 2)

                image = image[:, remove:image.shape[1] - remove, :]
                if use_masks:
                    masks = masks[:, remove:masks.shape[1] - remove, :]

            if do_preprocessing:
                if self.center_color_to_imagenet:
                    image = image.astype(np.float64)
                    image[..., 0] -= 123.68  # R
                    image[..., 1] -= 116.779  # G
                    image[..., 2] -= 103.939  # B
                else:
                    image /= 255.0

            # Append to batch
            batch_of_input_images.append(image)
            batch_of_target_masks.append(masks)
            batch_of_target_labels.append(labels)

            self.__cur_img_idx += 1

        if use_masks:
            batch_of_target_bboxes = np.zeros((1, batch_of_target_masks[0].shape[2], 4))
            for batch, mask_set in enumerate(batch_of_target_masks):

                batch_of_target_bboxes[batch, :mask_set.shape[2]] = extract_bboxes(mask_set)[:mask_set.shape[2]]

                for b, box in enumerate(batch_of_target_bboxes[batch]):
                    y1, x1, y2, x2 = box
                    batch_of_target_bboxes[batch][b] = np.array((x1, y1, x2, y2))
        else:
            batch_of_target_bboxes = batch_of_target_masks

        batch_of_input_images = np.asarray(batch_of_input_images, dtype=np.float32)

        return batch_of_input_images, batch_of_target_masks, batch_of_target_bboxes, batch_of_target_labels, self._num_classes

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        return self._get_x_y([index])

    def __next__(self):
        """
        Called when iterating over the dataset.

        :return:
        """

        idx = self.__cur_img_idx
        if self.__cur_img_idx >= len(self):
            idx = self.__cur_img_idx = 0

        return self[idx]

    def generate_image_id_map(self):
        for i, image in enumerate(self._images.items()):
            image_id, image_info = image
            self.__image_id_map[i] = image_id

    def get_label_names(self) -> List[str]:
        return self._label_names

    def _img_idx_to_id(self, idx: int):
        return self.__image_id_map[idx]

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

    def _load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        # image = skimage.io.imread(self.image_info[image_id]['path'])

        im = Image.open(self._images[image_id]['path'])

        sys.stdout.write('Loading image {}\n'.format(self._images[image_id]['path']))
        im = im.resize(
            (
                int(self._images[image_id]['width']),
                int(self._images[image_id]['height'])
            ),
            Image.ANTIALIAS
        )

        image = np.array(im, dtype=np.float32)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            import skimage
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        return image

    def load_mask(self, image_idx: int, as_box: bool = False) -> Tuple[np.array, np.array]:
        """
        Load a set of bitmap-masks for a given image.

        :param image_idx: Image index for which to load masks
        :type image_idx: int
        :return: Tuple of set of bitmap masks and classes
        :rtype: Tuple[np.array, np.array]
        """

        # Retrieve actual image id according to external dataset
        image_id = self.__image_id_map[image_idx]

        # Get mask data (label and polygon points in the form of (x1,y1,x2,y2,x3,y3,...,xn,yn)
        mask_data = self._masks.get(image_id)

        # Prepare the mask array
        masks = np.zeros([
            self._images.get(image_id).get('height'),
            self._images.get(image_id).get('width'),
            len(mask_data.get('masks_raw'))
        ], dtype=np.uint8) if not as_box else np.zeros((len(mask_data.get('masks_raw')), 4))

        labels = []

        for m_idx, mask_info in enumerate(mask_data.get('masks_raw')):

            # Get label, labels are already simplified when extracted (if necessary)
            labels.append(int(mask_info.get('category_id')))

            if as_box:
                x1, y1, w, h = mask_info.get('bbox')
                masks[m_idx] = np.multiply(np.array([x1, y1, x1 + w, y1 + h]), self.IMAGE_FACTOR)
            else:
                pts = mask_info.get('segmentation', [[]])[0]

                # Create bitmap from polygon
                masks[:, :, m_idx:m_idx + 1] = cv2.fillPoly(
                    masks[:, :, m_idx:m_idx + 1].copy(),
                    pts=np.array([[(int(pts[p] * self.IMAGE_FACTOR), int(pts[p + 1] * self.IMAGE_FACTOR)) for p in range(0, len(pts), 2)]], dtype=np.int32),
                    color=1
                )

        return masks.astype(np.int32), np.array(labels).astype(np.int32)

    def get_image_info(self) -> list:
        return self.__image_info

    def get_simplified_label(self, label):
        for key, cluster in self.CLASS_CLUSTERS.items():
            if label in cluster:
                return key

        return None

    def set_image_info(self, d: list):
        self.__image_info = d
        seen_img_ids = [info.get('id') for info in self.__image_info]
        del_ids = list(filter(lambda k: k not in seen_img_ids, self._images.keys()))
        for k in del_ids:
            del self._images[k]

    @abstractmethod
    def compile_dataset(self):
        raise NotImplementedError()

    @abstractmethod
    def register_image(self, group_name: str, image_id: int, path: str, width: int, height: int):
        raise NotImplementedError()

    @abstractmethod
    def register_label(self, group_name: str, label_id: int, label_name: str):
        raise NotImplementedError()

    @abstractmethod
    def load_image(self, image_idx: int) -> Image:
        raise NotImplementedError()

    @abstractmethod
    def get_xy(self, indices: List[int]):
        raise NotImplementedError()
