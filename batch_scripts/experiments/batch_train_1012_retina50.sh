#!/bin/bash

python main.py --purpose train    --env 1012-transfer-learning-image-augmentation-fine-tuning                  --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints > train-1012-retina50.log 2>&1
python main.py --purpose evaluate --env 1012-transfer-learning-image-augmentation-fine-tuning                  --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1012-transfer-learning-image-augmentation-fine-tuning-body-shots-eval  --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-body-shots-eval --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1012-transfer-learning-image-augmentation-fine-tuning-cases-eval       --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-cases-eval      --eval_images --eval_heatmaps

python main.py --purpose train    --env 1012b-transfer-learning-image-augmentation-fine-tuning                 --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints > train-1012b-retina50.log 2>&1
python main.py --purpose evaluate --env 1012b-transfer-learning-image-augmentation-fine-tuning                 --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1012b-transfer-learning-image-augmentation-fine-tuning-body-shots-eval --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-body-shots-eval --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1012b-transfer-learning-image-augmentation-fine-tuning-cases-eval      --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-cases-eval      --eval_images --eval_heatmaps

python main.py --purpose train    --env 1012c-transfer-learning-image-augmentation-fine-tuning                 --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints > train-1012c-retina50.log 2>&1
python main.py --purpose evaluate --env 1012c-transfer-learning-image-augmentation-fine-tuning                 --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1012c-transfer-learning-image-augmentation-fine-tuning-body-shots-eval --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-body-shots-eval --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1012c-transfer-learning-image-augmentation-fine-tuning-cases-eval      --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-cases-eval      --eval_images --eval_heatmaps