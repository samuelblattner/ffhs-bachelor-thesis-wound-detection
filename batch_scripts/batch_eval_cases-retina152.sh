#!/bin/bash

python main.py --purpose evaluate --env 1001-default-nets-all-cases-eval                                      --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1002-transfer-learning-all-cases-eval                                 --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1003-tl-fine-tuning-all-cases-eval                                    --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose evaluate --env 1010-image-augmentation-all-cases-eval                                --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1010b-image-augmentation-all-cases-eval                               --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1010c-image-augmentation-all-cases-eval                               --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose evaluate --env 1011-transfer-learning-image-augmentation-all-cases-eval              --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1011b-transfer-learning-image-augmentation-all-cases-eval             --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1011c-transfer-learning-image-augmentation-all-cases-eval             --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose evaluate --env 1012-transfer-learning-image-augmentation-fine-tuning-all-cases-eval  --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1012b-transfer-learning-image-augmentation-fine-tuning-all-cases-eval --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1012c-transfer-learning-image-augmentation-fine-tuning-all-cases-eval --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose evaluate --env 1012-transfer-learning-image-augmentation-fine-tuning-all-cases-eval  --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1012b-transfer-learning-image-augmentation-fine-tuning-all-cases-eval --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1012c-transfer-learning-image-augmentation-fine-tuning-all-cases-eval --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose evaluate --env 1020-close-up-wounds-all-cases-eval                                   --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1021-close-up-wounds-fine-tuning-all-cases-eval                       --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1030-joint-puppet-close-up-wounds-all-cases-eval                      --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1031-joint-puppet-close-up-wounds-fine-tuning-all-cases-eval          --gpu_no 1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
