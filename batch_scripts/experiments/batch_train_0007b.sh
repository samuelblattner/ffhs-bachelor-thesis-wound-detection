#/bin/bash

python main.py --purpose train --env 0007b-transfer-image-hires-augmentation-scale-1-7-rot-90-ep1000-lr0_0001 --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0007b-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0007b-transfer-image-hires-augmentation-scale-1-7-rot-90-ep1000-lr0_0001 --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection//evaluation/
python main.py --purpose evaluate --env 0007b-transfer-image-hires-augmentation-scale-1-7-rot-90-ep1000-lr0_0001 --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection//evaluation/
mkdir /home/blsa/projects/confidential/wound-detection/evaluation/neweval/0007b-transfer-image-hires-augmentation-scale-1-7-rot-90-ep1000-lr0_0001
cp /home/blsa/projects/confidential/wound-detection/evaluation/0007b-transfer-image-hires-augmentation-scale-1-7-rot-90-ep1000-lr0_0001/*.csv /home/blsa/projects/confidential/wound-detection/evaluation/neweval/0007b-transfer-image-hires-augmentation-scale-1-7-rot-90-ep1000-lr0_0001
