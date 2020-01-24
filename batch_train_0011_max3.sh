#/bin/bash

# 0011
python main.py --purpose train --env 0011-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011-max3-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0011-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 0011-max3-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

#python main.py --purpose train --env 0011b-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011b-max3-retina152-train.log 2>&1
#python main.py --purpose evaluate --env 0011b-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 0011b-max3-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

#python main.py --purpose train --env 0011c-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011c-max3-retina152-train.log 2>&1
#python main.py --purpose evaluate --env 0011c-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 0011c-max3-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

#python main.py --purpose train --env 0011d-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0011d-max3-retina152-train.log 2>&1
#python main.py --purpose evaluate --env 0011d-max3-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 0011d-max3-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/


python main.py --purpose evaluate --env 0011-max3-all-cases-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
python main.py --purpose evaluate --env 0011-max3-all-cases-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/

#python main.py --purpose evaluate --env 0011b-max3-all-cases-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#python main.py --purpose evaluate --env 0011b-max3-all-cases-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#
#python main.py --purpose evaluate --env 0011c-max3-all-cases-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#python main.py --purpose evaluate --env 0011c-max3-all-cases-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#
#python main.py --purpose evaluate --env 0011d-max3-all-cases-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#python main.py --purpose evaluate --env 0011d-max3-all-cases-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/

python main.py --purpose evaluate --env 0011-max3-all-cases-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
python main.py --purpose evaluate --env 0011-max3-all-cases-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/

#python main.py --purpose evaluate --env 0011b-max3-all-cases-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#python main.py --purpose evaluate --env 0011b-max3-all-cases-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#
#python main.py --purpose evaluate --env 0011c-max3-all-cases-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#python main.py --purpose evaluate --env 0011c-max3-all-cases-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#
#python main.py --purpose evaluate --env 0011d-max3-all-cases-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#python main.py --purpose evaluate --env 0011d-max3-all-cases-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/


# 0014
#python main.py --purpose train --env 0014-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0014-max3-retina152-train.log 2>&1
#python main.py --purpose evaluate --env 0014-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 0014-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

#python main.py --purpose train --env 0014b-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0014b-max3-retina152-train.log 2>&1
#python main.py --purpose evaluate --env 0014b-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 0014b-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

python main.py --purpose train --env 0014c-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0014c-max3-retina152-train.log 2>&1
python main.py --purpose evaluate --env 0014c-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 0014c-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/

#python main.py --purpose train --env 0014d-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ > 0014d-max3-retina152-train.log 2>&1
#python main.py --purpose evaluate --env 0014d-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
#python main.py --purpose evaluate --env 0014d-max3-all-cases-transfer-imgaug-puppetbase --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/


#python main.py --purpose evaluate --env 0014-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#python main.py --purpose evaluate --env 0014-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#
#python main.py --purpose evaluate --env 0014b-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#python main.py --purpose evaluate --env 0014b-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/

python main.py --purpose evaluate --env 0014c-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
python main.py --purpose evaluate --env 0014c-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/

#python main.py --purpose evaluate --env 0014d-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#python main.py --purpose evaluate --env 0014d-max3-all-cases-transfer-imgaug-puppetbase-all-cases --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/all-cases/
#
#python main.py --purpose evaluate --env 0014-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#python main.py --purpose evaluate --env 0014-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#
#python main.py --purpose evaluate --env 0014b-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#python main.py --purpose evaluate --env 0014b-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/

python main.py --purpose evaluate --env 0014c-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
python main.py --purpose evaluate --env 0014c-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/

#python main.py --purpose evaluate --env 0014d-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/
#python main.py --purpose evaluate --env 0014d-max3-all-cases-transfer-imgaug-puppetbase-body-shots-eval --gpu_no 2 --full_size_eval true --net_type retina-resnet152 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/body-shots/