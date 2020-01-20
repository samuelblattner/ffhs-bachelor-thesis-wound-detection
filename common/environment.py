import json
import sys
from typing import Tuple, Type, List

import imgaug
import os

from os.path import join

from common.adapters.datasets.interfaces import AbstractDataset
from common.adapters.datasets.union import UnionDataset
from common.enums import NeuralNetEnum, ModelPurposeEnum
from neural_nets.retina_net.keras_retinanet.preprocessing.generator import Generator

AUGMENTATION_MAP = {
    'fliplr': imgaug.augmenters.Fliplr,
    'flipud': imgaug.augmenters.Flipud,
    'affine': imgaug.augmenters.Affine,
    'grayscale': imgaug.augmenters.Grayscale,
    'sequential': imgaug.augmenters.Sequential,
    'crop': imgaug.augmenters.CropToFixedSize
}

CLASS_NAMES = {
    'simplified': {
        1: 'Sharp Force',
        2: 'Blunt Force'
    },
    'complete': {

    }
}


class Environment:
    #: Name for the environment
    name: str = None

    #: Description
    description: str = None

    #: Batch size
    batch_size: int = 1

    #: Num epochs
    epochs: int = 1

    #: Max size of longer image side
    max_image_side_length: int = 768

    #: Whether to use transfer learning (i.e. pre-trained feature/backbone weights) or not
    use_transfer_learning: bool = False

    #: Whether to freeze base layers or allow for them to be trained
    allow_base_layer_training: bool = False

    #: Whether to train on high-res wound image patches (instead of full body shots)
    extract_wound_patches: bool = False

    # LEGACY
    # ----------------------------------
    #: Name for the train dataset
    train_dataset_name: str = None
    val_dataset_name: str = None
    test_dataset_name: str = None

    #: Dataset split
    dataset_split = None
    # ----------------------------------

    datasets: List = []

    #: SHuffle
    shuffle_dataset: bool = True

    #: Shuffle seed. Set to make dataset shuffling deterministic
    shuffle_seed: int = 0

    #: Kind of neural net
    neural_net_type: NeuralNetEnum = NeuralNetEnum.MRCNN_RESNET50

    #: Purpose of the net to be created
    purpose: ModelPurposeEnum = ModelPurposeEnum.TRAINING

    #: Learning Rate
    learning_rate: float = 0.0001

    #: Data Augmentation
    augmentation = None

    #: Simplifies all classes into only two main classes
    simplify_classes: bool = True

    #: Normalize color
    center_color_to_imagenet: bool = False

    #: Inflated
    valid: bool = False
    prepared: bool = False

    img_scale_mode: str = 'square'
    class_names: list = []

    checkpoint_root: str = './checkpoints'
    data_root: str = './data'

    dataset_class: Type[AbstractDataset] = None

    __train_dataset: AbstractDataset = None
    __val_dataset: AbstractDataset = None
    __test_dataset: AbstractDataset = None

    __num_classes: int = 13

    evaluation_dir: str = './evaluation'
    pre_image_scale: float = 0.5
    split_by_filename_base: bool = False
    max_examples_per_filename_base: int = 0
    gpu_no: int = 0
    full_size_eval: bool = False
    eval_images: bool = False

    @classmethod
    def from_json(cls, path: str):
        """
        Load a env from a JSON file

        :param path: Path to json file
        :type path: str
        :return: Environment configuration loaded from JSON file
        :rtype Environment
        """

        try:
            with open(path, 'r', encoding='utf-8') as file:
                config_dict = json.loads(''.join(file.readlines()))

                env = Environment()

                env.name = config_dict.get('name', env.name)
                env.description = config_dict.get('description', env.description)
                env.batch_size = config_dict.get('batch_size', env.batch_size)
                env.epochs = config_dict.get('epochs', env.epochs)
                env.max_image_side_length = config_dict.get('max_image_side_length', env.max_image_side_length)
                env.use_transfer_learning = config_dict.get('use_transfer_learning', env.use_transfer_learning)
                env.allow_base_layer_training = config_dict.get('allow_base_layer_training', env.allow_base_layer_training)
                env.extract_wound_patches = config_dict.get('extract_wound_patches', env.extract_wound_patches)
                env.train_dataset_name = config_dict.get('train_dataset_name', env.train_dataset_name)
                env.val_dataset_name = config_dict.get('val_dataset_name', env.val_dataset_name)
                env.test_dataset_name = config_dict.get('test_dataset_name', env.test_dataset_name)
                env.dataset_split = config_dict.get('dataset_split', env.dataset_split)
                env.shuffle_dataset = config_dict.get('shuffle_dataset', env.shuffle_dataset)
                env.shuffle_seed = config_dict.get('shuffle_seed', env.shuffle_seed)
                env.learning_rate = config_dict.get('learning_rate', env.learning_rate)
                env.augmentation = config_dict.get('augmentation', env.augmentation)
                env.simplify_classes = config_dict.get('simplify_classes', env.simplify_classes)
                env.center_color_to_imagenet = config_dict.get('center_color_to_imagenet', env.center_color_to_imagenet)
                env.pre_image_scale = config_dict.get('pre_image_scale', env.pre_image_scale)
                env.split_by_filename_base = config_dict.get('split_by_filename_base', env.split_by_filename_base)
                env.max_examples_per_filename_base = config_dict.get('max_examples_per_filename_base', env.max_examples_per_filename_base)
                env.gpu_no = config_dict.get('gpu_no', env.gpu_no)
                env.full_size_eval = config_dict.get('full_size_eval', env.full_size_eval)
                env.data_root = config_dict.get('data_dir', env.data_root)
                env.datasets = config_dict.get('datasets', [])

                return env
        except FileNotFoundError:
            sys.stderr.write('No env \'{}\' found!\n'.format(path))
            exit(1)

    @property
    def full_config_name(self) -> str:
        return self.name.replace(' ', '_')

    def replace_tuples(self, params):
        for key, val in params.items():
            if type(val) == str and val[0] == '(' and val[-1] == ')':
                from ast import literal_eval as make_tuple
                params[key] = make_tuple(val)
            elif type(val) == dict:
                self.replace_tuples(val)

    def inflate_augmentation(self, params: dict):
        aug_class = AUGMENTATION_MAP.get(params.get('type'))
        if aug_class is None:
            return None
        sub_params = params.get('params', {})
        self.replace_tuples(sub_params)
        args = []
        for key, val in sub_params.items():
            if key == '_':
                args.append(val)
                continue
            if isinstance(val, List):
                for i, item in enumerate(val):
                    sub_params[key][i] = self.inflate_augmentation(sub_params[key][i])

        if '_' in sub_params:
            del sub_params['_']

        return aug_class(*args, **sub_params)

    def prepare(self):
        if not self.valid:
            self.validate()

        datasets = self.datasets if len(self.datasets) > 0 else [{
            'name': self.train_dataset_name,
            'split': self.dataset_split,
            'pre_image_scale': self.pre_image_scale,
            'split_by_filename_base': self.split_by_filename_base,
            'max_examples_per_filename_base': self.max_examples_per_filename_base,
            'augmentation': self.augmentation
        }]

        train_datasets = []
        val_datasets = []
        test_datasets = []

        for dataset in datasets:
            # Transformations
            augmentation = self.inflate_augmentation(dataset.get('augmentation', {}))
            # if self.augmentation and type(self.augmentation) == dict:
            #     self.augmentation = self.inflate_augmentation(self.augmentation)
            train_dataset, val_dataset, test_dataset = self.dataset_class.create_datasets(
                train_dataset_path=join(self.data_root, dataset.get('name')),
                dataset_split=dataset.get('split'),
                shuffle=self.shuffle_dataset,
                shuffle_seed=self.shuffle_seed,
                batch_size=self.batch_size,
                max_image_side_length=self.max_image_side_length,
                augmentation=augmentation,
                center_color_to_imagenet=self.center_color_to_imagenet,
                simplify_classes=self.simplify_classes,
                image_scale_mode=self.img_scale_mode,
                pre_image_scale=self.pre_image_scale,
                split_by_filename_base=dataset.get('split_by_filename_base'),
                max_examples_per_filename_base=dataset.get('max_examples_per_filename_base', 0)
            )

            train_dataset.IMAGE_FACTOR = val_dataset.IMAGE_FACTOR = test_dataset.IMAGE_FACTOR = dataset.get('pre_image_scale', 0.5)

            self.class_names = train_dataset.get_label_names()

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_datasets.append(test_dataset)

        self.__train_dataset = UnionDataset(train_datasets)
        self.__val_dataset = UnionDataset(val_datasets)
        self.__test_dataset = UnionDataset(test_datasets)

    def get_datasets(self) -> Tuple[AbstractDataset, AbstractDataset, AbstractDataset]:
        return self.__train_dataset, self.__val_dataset, self.__test_dataset

    def validate(self):
        """
        Validates the env and raises an exception if invalid.
        """

        # Check dataset class
        if self.dataset_class is None:
            raise ValueError('You need to set a dataset class!')

        # Check name
        if self.name is None or self.name == '':
            raise ValueError('The environment requires a name. No name was set!')

        # Check batch size
        if self.batch_size <= 0:
            raise ValueError('Please specify a batch size >= 1. Current value is: {}'.format(self.batch_size))

        # Check epochs
        if self.epochs <= 0:
            raise ValueError('Please specify an epoch number of >= 1. Current value is: {}'.format(self.epochs))

        # Check image side length
        if self.max_image_side_length <= 0:
            raise ValueError('Please specify a max image side number of >= 1. Current value is: {}'.format(self.max_image_side_length))

        # Check datasets
        if self.purpose == ModelPurposeEnum.TRAINING:
            if len(self.datasets) <= 0 and self.train_dataset_name is None:
                raise ValueError('Please specify at least a training dataset name if you intend to train the model.')

            for dataset_name in (self.train_dataset_name, self.val_dataset_name, self.test_dataset_name):
                if dataset_name is None:
                    continue
                if not os.path.isdir(join(self.data_root, dataset_name)):
                    raise ValueError('No dataset by the name \'{dataset_name}\' found in directory {dir}!'.format(
                        dataset_name=dataset_name,
                        dir=self.data_root
                    ))

        # Check split
        if self.dataset_split and sum(self.dataset_split) != 1.0:
            raise ValueError('Expected split ratios to add up to 1, but sum was {sum}. Please make sure the train, val, test ratios add up to 1!'.format(
                sum=sum(self.dataset_split)))

        if self.purpose not in ModelPurposeEnum:
            raise ValueError('Field \'purpose\' contains an invalid value. Please use the ModelPurposeEnum to assign a value!')

        # Check augmentation
        if self.augmentation and type(self.augmentation) == dict and self.augmentation.get('type') not in AUGMENTATION_MAP:
            raise ValueError('Unknown augmentation \'{key}\'.'.format(key=self.augmentation.get('type')))

        self.valid = True
