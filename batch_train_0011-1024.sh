#/bin/bash

python main.py --purpose train --env 0011-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 0011-all-cases --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose train --env 0011b-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011b-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011b-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 0011b-all-cases --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose train --env 0011c-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011c-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011c-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 0011c-all-cases --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose train --env 0011d-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011d-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011d-all-cases --gpu_no 0 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 0011d-all-cases --gpu_no 0 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
