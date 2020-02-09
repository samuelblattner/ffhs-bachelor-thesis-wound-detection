#!/bin/bash

python main.py --purpose train --env 0022a-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0022-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0022a-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_images
# python main.py --purpose evaluate --env 0022a-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0022b-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0022b-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0022b-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
# python main.py --purpose evaluate --env 0022b-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0022c-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0022c-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0022c-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
# python main.py --purpose evaluate --env 0022c-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0022d-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0022d-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0022d-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
# python main.py --purpose evaluate --env 0022d-transfer-imgaug-scale-1-2-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
