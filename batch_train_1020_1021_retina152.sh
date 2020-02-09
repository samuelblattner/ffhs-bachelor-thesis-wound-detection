#!/bin/bash

python main.py --purpose train    --env 1020-close-up-wounds                                       --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints > train-1020-retina152.log 2>&1
python main.py --purpose evaluate --env 1020-close-up-wounds                                       --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1020-close-up-wounds-body-shots-eval                       --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-body-shots-eval --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1020-close-up-wounds-cases-eval                            --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-cases-eval      --eval_images --eval_heatmaps

mkdir -p /checkpoints/1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
cp /checkpoints/1020a_Close_Up_Wounds--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
mv /checkpoints/1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/1021a_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152_0001.h5

mkdir -p /checkpoints/1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
cp /checkpoints/1020_Close_Up_Wounds--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
mv /checkpoints/1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/1021b_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152_0001.h5

mkdir -p /checkpoints/1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
cp /checkpoints/1020c_Close_Up_Wounds--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
mv /checkpoints/1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/1021c_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152_0001.h5

mkdir -p /checkpoints/1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
cp /checkpoints/1020d_Close_Up_Wounds--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152
mv /checkpoints/1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /checkpoints/1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152/1021d_Close_Up_Wounds_Fine_Tuning--RetinaNet-Resnet152_0001.h5

python main.py --purpose train    --env 1021-close-up-wounds-fine-tuning                           --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints > train-1021-retina152.log 2>&1
python main.py --purpose evaluate --env 1021-close-up-wounds-fine-tuning                           --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1021-close-up-wounds-fine-tuning-body-shots-eval           --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-body-shots-eval --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1021-close-up-wounds-fine-tuning-cases-eval                --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation --eval_name_suffix=-cases-eval      --eval_images --eval_heatmaps