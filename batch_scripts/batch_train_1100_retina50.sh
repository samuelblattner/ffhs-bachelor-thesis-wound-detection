#!/bin/bash

python main.py --purpose train --batch_size=4    --env 1100-cases  --start_from_xval_k=1          --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ > train_1100_retina50.log 2>&1
python main.py --purpose evaluate --env 1100-cases                                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1100-cases-body-shots-eval                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100a_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100b_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100c_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100d_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train --batch_size=3    --env 1101-cases-fine-tuning                     --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1101_retina50.log 2>&1
python main.py --purpose evaluate --env 1101-cases-fine-tuning                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1101-cases-fine-tuning-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval
