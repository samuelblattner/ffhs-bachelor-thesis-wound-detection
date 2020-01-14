#/bin/bash

python main.py --purpose train --env 0017e-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017e-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017e-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017e-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017f-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017f-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017f-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017f-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017g-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017g-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017g-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017g-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/

python main.py --purpose train --env 0017h-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0017h-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0017h-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
python main.py --purpose evaluate --env 0017h-TL-ImgAug-rot90-closeup-wounds-conf --gpu_no 1 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/evaluation/
