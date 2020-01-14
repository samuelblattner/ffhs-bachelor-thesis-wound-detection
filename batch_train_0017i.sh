#/bin/bash

python main.py --purpose train --env 0017i-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017i-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017i-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017i-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017j-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017j-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017j-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017j-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017k-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017k-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017k-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017k-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017l-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017l-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017l-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017l-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
