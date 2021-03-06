#!/bin/bash

python main.py --purpose train --env 0014-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014-max6-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0014b-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014b-max6-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014b-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014b-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0014c-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014c-max6-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014c-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014c-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/

python main.py --purpose train --env 0014d-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014d-max6-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014d-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014d-max6-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
