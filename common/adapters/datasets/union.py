from typing import List, Union

from PIL import Image

from common.adapters.datasets.interfaces import AbstractDataset
from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator


class UnionDataset(Generator):
    """
    combines multiple datasets
    """

    datasets: List[Union[AbstractDataset, Generator]] = []

    def __init__(self, datasets: List[Union[AbstractDataset, Generator]]):
        self.datasets = datasets
        super(UnionDataset, self).__init__()

    def size(self):
        return sum([d.size() for d in self.datasets])

    def num_classes(self):
        return len(set([d.get_label_names() for d in self.datasets]))

    def has_label(self, label):
        return any([d.has_label(label) for d in self.datasets])

    def has_name(self, name):
        return any([d.has_name(name) for d in self.datasets])

    def name_to_label(self, name):
        pass

    def label_to_name(self, label):
        pass

    def image_aspect_ratio(self, image_index):
        pass

    def image_path(self, image_index):
        pass

    def load_image(self, image_index):
        pass

    def load_annotations(self, image_index):
        pass

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        return self.get_x_y(group)