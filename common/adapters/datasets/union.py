from itertools import chain
from typing import List, Union, Tuple
import numpy as np

from common.adapters.datasets.interfaces import AbstractDataset
from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator
from neural_nets.retina_net.keras_retinanet.utils.anchors import anchor_targets_bbox, guess_shapes
from neural_nets.retina_net.keras_retinanet.utils.image import TransformParameters, preprocess_image


class UnionDataset(AbstractDataset, Generator):
    """
    combines multiple datasets
    """

    def compile_dataset(self):
        self.group_method = 'random'
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

    def get_xy(self, indices: List[int]):
        pass

    def get_image_info(self) -> list:
        return list(chain(*[d.get_image_info() for d in self.datasets]))

    def load_mask(self, image_idx: int, as_box: bool = False) -> Tuple[np.array, np.array]:
        dataset, local_idx = self.__get_dataset_for_idx(image_idx)
        return dataset.load_mask(local_idx, as_box)

    datasets: List[Union[AbstractDataset, Generator]] = []

    def __init__(self, datasets: List[Union[AbstractDataset, Generator]]):
        self.datasets = datasets
        super(UnionDataset, self).__init__(
            dataset_path='',
            simplify_classes=False
        )
        self.compile_dataset()

    def __len__(self):
        return self.size()

    def __get_dataset_for_idx(self, idx: int) -> Tuple[AbstractDataset, int]:

        sum_len = 0
        for dataset in self.datasets:

            if idx < sum_len + len(dataset):
                return dataset, idx - sum_len

            sum_len += len(dataset)

    def size(self):
        return sum([d.size() for d in self.datasets])

    def num_classes(self):
        return len(set(chain(*[d.get_label_names() for d in self.datasets])))

    def has_label(self, label):
        return any([d.has_label(label) for d in self.datasets])

    def has_name(self, name):
        return any([d.has_name(name) for d in self.datasets])

    def name_to_label(self, name):
        return self.datasets[0].name_to_label(name)

    def label_to_name(self, label):
        return self.datasets[0].label_to_name(label)

    def image_aspect_ratio(self, image_index):
        dataset, local_idx = self.__get_dataset_for_idx(image_index)
        return dataset.image_aspect_ratio(local_idx)

    def image_path(self, image_index):
        dataset, local_idx = self.__get_dataset_for_idx(image_index)
        return dataset.image_path(local_idx)

    def load_image(self, image_index):
        dataset, local_idx = self.__get_dataset_for_idx(image_index)
        return dataset.load_image(local_idx)

    def load_annotations(self, image_index):
        dataset, local_idx = self.__get_dataset_for_idx(image_index)
        return dataset.load_annotations(local_idx)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        # group = self.groups[index]
        dataset, local_idx = self.__get_dataset_for_idx(index)
        return dataset.__getitem__(local_idx)
