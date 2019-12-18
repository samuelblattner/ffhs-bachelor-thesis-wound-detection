from typing import List

from imgaug.augmenters import Augmenter, np
from mrcnn.utils import Dataset

from common.datasets.dataset import BaseDataset


class MRCNNDataset(BaseDataset, Dataset):

    def __init__(self, dataset_path: str, simplify_classes: bool = False, batch_size: int = 1, max_image_side_length: int = 512,
                 augmentation: Augmenter = None, center_color_to_imagenet: bool = False, image_scale_mode: str = 'square', pre_image_scale=0.5):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [] # [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        super(MRCNNDataset, self).__init__(dataset_path, simplify_classes, batch_size, max_image_side_length, augmentation, center_color_to_imagenet,
                                           image_scale_mode, pre_image_scale )

    def register_label(self, group_name: str, label_id: int, label_name: str):
        self.add_class(
            group_name,
            label_id,
            label_name
        )

    def compile_dataset(self):
        self.prepare()

    def register_image(self, group_name: str, image_id: int, path: str, width: int, height: int):
        self.add_image(
            source='puppet',
            image_id=image_id,
            path=path,
            width=width,
            height=height,
        )

    def load_image(self, image_idx):
        img = np.asarray(self._load_image(self._img_idx_to_id(image_idx)))
        return img/127.5-1.

    def get_image_info(self) -> list:
        return self.image_info

    def _get_x_y(self, indices: List[int]):
        batch_of_input_images, batch_of_mask_sets, batch_of_label_sets, num_labels = super(MRCNNDataset, self)._get_x_y(indices)
        return np.asarray(batch_of_input_images), np.asarray(batch_of_mask_sets)

    def set_image_info(self, d: list):
        super(MRCNNDataset, self).set_image_info(d)
        self.image_info = d
        # seen_img_ids = [info.get('id') for info in self.image_info]
        # del_ids = list(filter(lambda k: k not in seen_img_ids, self.image.keys()))
        # for k in del_ids:
        #     del self._images[k]