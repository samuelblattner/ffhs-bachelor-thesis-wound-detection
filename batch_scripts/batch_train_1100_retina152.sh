#!/bin/bash

# 1100, 1101
# ==========
#python main.py --purpose train    --env 1100-cases                       --gpu_no 3 --batch_size=1 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ > train_1100_retina152.log 2>&1
#python main.py --purpose evaluate --env 1100-cases                                                --gpu_no 3 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 1100-cases-body-shots-eval                                --gpu_no 3 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval
#
#find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
#find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.pdf' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/roc-curves \;
#git add evaluation/final/* && git commit -m "Add 1100 evals RetinaNet 152" && git push
#
#
#
#mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet152
#mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet152
#mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet152
#mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet152
#
#cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100a_Cases--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet152
#cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100b_Cases--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet152
#cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100c_Cases--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet152
#cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100d_Cases--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet152
#
#mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet152/1101a_Cases_Fine_Tuning--RetinaNet-Resnet152_0001.h5
#mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet152/1101b_Cases_Fine_Tuning--RetinaNet-Resnet152_0001.h5
#mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet152/1101c_Cases_Fine_Tuning--RetinaNet-Resnet152_0001.h5
#mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet152/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet152/1101d_Cases_Fine_Tuning--RetinaNet-Resnet152_0001.h5

python main.py --purpose train --batch_size=1    --env 1101-cases-fine-tuning --loss_patience=5 --val_loss_patience=5                    --gpu_no 3 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1101_retina152.log 2>&1
python main.py --purpose evaluate --env 1101-cases-fine-tuning                                    --gpu_no 3 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1101-cases-fine-tuning-body-shots-eval                    --gpu_no 3 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.pdf' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/roc-curves \;
git add evaluation/final/* && git commit -m "Add 1101 evals RetinaNet 152" && git push
