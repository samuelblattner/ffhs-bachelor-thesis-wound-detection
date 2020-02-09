#!/bin/bash

python main.py --purpose train    --env 1112-cases-multishot-6                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1112-cases-multishot-6                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1112-cases-multishot-6-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112a_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112b_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112c_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112d_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train    --env 1113-cases-multishot-6-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1113-cases-multishot-6-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1113-cases-multishot-6-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/