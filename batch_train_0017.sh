#/bin/bash

python main.py --purpose train --env 0017-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017b-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017b-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017b-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017b-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017c-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017c-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017c-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017c-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017d-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017d-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017d-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017d-TL-ImgAug-1x-3x-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
