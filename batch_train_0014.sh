#/bin/bash

python main.py --purpose train --env 0014-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014-all-cases-transfer-imgaug-puppetbase
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0014_All_Cases_Transfer_ImgAug_Puppetbase/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014-all-cases-transfer-imgaug-puppetbase

python main.py --purpose train --env 0014b-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014b-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014b-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014b-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014b-all-cases-transfer-imgaug-puppetbase
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0014_All_Cases_Transfer_ImgAug_Puppetbase/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014b-all-cases-transfer-imgaug-puppetbase

python main.py --purpose train --env 0014c-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014c-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014c-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014c-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014c-all-cases-transfer-imgaug-puppetbase
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0014_All_Cases_Transfer_ImgAug_Puppetbase/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014c-all-cases-transfer-imgaug-puppetbase

python main.py --purpose train --env 0014d-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ > 0014d-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014d-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
python main.py --purpose evaluate --env 0014d-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --checkpoint_dir=/home/blsa/checkpoints/ --eval_dir=/home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/
mkdir /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014d-all-cases-transfer-imgaug-puppetbase
cp /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/confidential/0014_All_Cases_Transfer_ImgAug_Puppetbase/*.csv /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/neweval/0014d-all-cases-transfer-imgaug-puppetbase
