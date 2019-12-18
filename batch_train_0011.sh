#/bin/bash

python main.py --purpose train --env 0011-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0011-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0011-all-cases --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011-all-cases
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0011_All_Cases/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011-all-cases

python main.py --purpose train --env 0011b-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0011b-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011b-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0011b-all-cases --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011b-all-cases
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0011_All_Cases/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011b-all-cases

python main.py --purpose train --env 0011c-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0011c-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011c-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0011c-all-cases --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011c-all-cases
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0011_All_Cases/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011c-all-cases

python main.py --purpose train --env 0011d-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0011d-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011d-all-cases --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0011d-all-cases --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011d-all-cases
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0011_All_Cases/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0011d-all-cases
