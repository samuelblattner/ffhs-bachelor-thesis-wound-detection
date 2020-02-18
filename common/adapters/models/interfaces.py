import os
import re
import sys
from abc import ABCMeta, abstractmethod
from os.path import join
from typing import Tuple, List

import numpy as np

from tensorflow.python.keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.callbacks import TensorBoard

from common.detection import Detection
from common.enums import ModelPurposeEnum
from common.environment import Environment


class AbstractModelAdapter:
    __metaclass__ = ABCMeta

    #: Model used for training
    train_model: Model

    #: Model used for prediction
    inference_model: Model

    #: Environment for training and prediction
    env: Environment = None

    def __init__(self, environment: Environment):
        self.env = environment
        self.train_model, self.inference_model = self.build_models()
        self.load_latest_checkpoint()

    @property
    def full_name(self) -> str:
        return '{}--{}'.format(self.env.name, self.get_name()).replace(' ', '_')

    @property
    def num_classes(self) -> int:
        return 2 if self.env.simplify_classes else 13

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        checkpoint_dir_path, checkpoint_path, latest_checkpoint_path = self.get_checkpoint_location()

        dir_names = next(os.walk(self.env.checkpoint_root))[1]
        key = self.full_name
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(checkpoint_dir_path))
        # Pick last directory
        dir_name = os.path.join(self.env.checkpoint_root, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(self.full_name), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])

        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        print('--> Loading weights from ', filepath)
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        print('Loading weights for ', 'train ' if self.env.purpose == ModelPurposeEnum.TRAINING else 'inference')
        keras_model = self.train_model if self.env.purpose == ModelPurposeEnum.TRAINING else self.inference_model

        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def get_checkpoint_location(self) -> Tuple[str, str, str]:
        checkpoint_dir_path = join(self.env.checkpoint_root, self.full_name)
        checkpoint_path = join(checkpoint_dir_path, "{}_{{epoch:04d}}.h5".format(self.full_name))
        checkpoint_latest_path = join(checkpoint_dir_path, "{}_latest.h5".format(self.full_name))
        return checkpoint_dir_path, checkpoint_path, checkpoint_latest_path

    def load_latest_checkpoint(self):

        try:
            last_checkpoint = self.find_last()
            self.load_weights(last_checkpoint, by_name=True)
            regex = r".*(\d{4})\.h5"
            m = re.match(regex, last_checkpoint)
            if m:
                # now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                #                         int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.train_model.epoch = int(m.group(1)) - 1 + 1
                print('----> Re-starting from epoch %d' % self.train_model.epoch)
        except FileNotFoundError:
            pass
        except StopIteration:
            pass
        except BaseException as e:
            print(e)

    def get_callbacks(self, loss_patience=15, val_loss_patience=30) -> List[Callback]:

        checkpoint_dir_path, checkpoint_path, checkpoint_latest_path = self.get_checkpoint_location()

        return [
            TensorBoard(
                log_dir=checkpoint_dir_path,
                batch_size=self.env.batch_size,
                write_images=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=True,
                save_best_only=True,
                monitor='val_loss',
            ),
            ModelCheckpoint(
                filepath=checkpoint_latest_path,
                verbose=True,
                save_best_only=True,
                monitor='val_loss',
            ),
            EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=val_loss_patience,
                verbose=1,
                mode='auto',
            ),
            EarlyStopping(
                monitor='loss',
                min_delta=0,
                patience=loss_patience,
                verbose=1,
                mode='auto',
            )
        ]

    def train(self, loss_patience=15, val_loss_patience=30, start_from_xval_k: int = None):
        """
        Train the network
        :return:
        """

        base_env_name = self.env.name

        for i, datasets in enumerate(self.env.iter_datasets()):

            if start_from_xval_k is not None and i < start_from_xval_k:
                continue

            if self.env.auto_xval:

                if self.env.x_val_auto_env_name:
                    self.env.name = re.sub(r'^(\d{4})', r'\1{}'.format('abcdefghijklmnopqrstuvwxyz'[i]), base_env_name)
                sys.stdout.write('--> X-Val: Training model {} of {} ({}):'.format(i, self.env.k_fold_x_val, self.env.name))

            self.train_model, self.inference_model = self.build_models()
            self.load_latest_checkpoint()

            train_dataset, val_dataset, test_dataset = datasets

            assert val_dataset.augmentation is None
            assert test_dataset.augmentation is None

            self.train_model.fit_generator(
                generator=train_dataset,
                steps_per_epoch=np.ceil(train_dataset.size() / self.env.batch_size),
                epochs=self.env.epochs,
                initial_epoch=self.train_model.epoch,
                verbose=1,
                validation_data=val_dataset,
                validation_steps=np.ceil(val_dataset.size() / self.env.batch_size),
                max_queue_size=5,
                workers=4,
                use_multiprocessing=False,
                shuffle=False,
                callbacks=self.get_callbacks(loss_patience, val_loss_patience)
            )

    @abstractmethod
    def predict(self, images, min_score=0.5) -> List[List[Detection]]:
        raise NotImplementedError()

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def build_models(self) -> Tuple[Model, Model]:
        raise NotImplementedError()

    @abstractmethod
    def generate_inference_heatmaps(self, image: np.array, plots) -> np.array:
        raise NotImplementedError()
