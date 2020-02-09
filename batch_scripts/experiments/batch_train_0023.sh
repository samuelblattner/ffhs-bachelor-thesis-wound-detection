#!/bin/bash

python main.py --purpose train --env 0023a-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0023-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0023a-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_images
# python main.py --purpose evaluate --env 0023a-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0023b-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0023b-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0023b-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
# python main.py --purpose evaluate --env 0023b-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0023c-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0023c-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0023c-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
# python main.py --purpose evaluate --env 0023c-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0023d-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --data_dir=data/vanilla_datasets > 0023d-retina152-train.log 2>&1
# python main.py --purpose evaluate --env 0023d-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
# python main.py --purpose evaluate --env 0023d-transfer-tbb-imgaug-scale-1-2-rot180-closeups --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/m-persist/tensorflow/puppet/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
