#!/bin/bash

# 1300, 1301
# ==========
python main.py --purpose train --batch_size=3    --env 1300-joint-cases-closeup-wounds                     --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1300_retina50.log 2>&1
python main.py --purpose evaluate --env 1300-joint-cases-closeup-wounds                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1300-joint-cases-closeup-wounds-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1300 evals RetinaNet 50" && git push


mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1300a_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1300b_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1300c_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1300d_Joint_Cases_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1301a_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1301b_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1301c_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1301d_Joint_Cases_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train --batch_size=3    --env 1301-joint-cases-closeup-wounds-fine-tuning         --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1301_retina50.log 2>&1
python main.py --purpose evaluate --env 1301-joint-cases-closeup-wounds-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1301-joint-cases-closeup-wounds-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1301 evals RetinaNet 50" && git push



# 1200, 1201
# ==========
python main.py --purpose train --batch_size=3    --env 1200-joint-cases-closeup-wounds-conf-only                     --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1200_retina50.log 2>&1
python main.py --purpose evaluate --env 1200-joint-cases-closeup-wounds-conf-only                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1200-joint-cases-closeup-wounds-conf-only-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1200 evals RetinaNet 50" && git push

mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1200a_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1200b_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1200c_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1200d_Joint_Cases_Closeup_Wounds_Conf_Only--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/1201a_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/1201b_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/1201c_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50/1201d_Joint_Cases_Closeup_Wounds_Conf_Only_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train --batch_size=3    --env 1201-joint-cases-closeup-wounds-conf-only-fine-tuning         --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1201_retina50.log 2>&1
python main.py --purpose evaluate --env 1201-joint-cases-closeup-wounds-conf-only-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1201-joint-cases-closeup-wounds-conf-only-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1201 evals RetinaNet 50" && git push




# 1112, 1113
# ==========
python main.py --purpose train --batch_size=3    --env 1112-cases-multishot-6                     --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1112_retina50.log 2>&1
python main.py --purpose evaluate --env 1112-cases-multishot-6                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1112-cases-multishot-6-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1112 evals RetinaNet 50" && git push


mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112a_Cases_Multishot_6--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112b_Cases_Multishot_6--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112c_Cases_Multishot_6--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1112d_Cases_Multishot_6--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train --batch_size=3    --env 1113-cases-multishot-6-fine-tuning         --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1113_retina50.log 2>&1
python main.py --purpose evaluate --env 1113-cases-multishot-6-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1113-cases-multishot-6-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1113 evals RetinaNet 50" && git push



# 1110, 1111
# ==========
python main.py --purpose train --batch_size=3    --env 1110-cases-multishot-3                     --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1110_retina50.log 2>&1
python main.py --purpose evaluate --env 1110-cases-multishot-3                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1110-cases-multishot-3-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1110 evals RetinaNet 50" && git push

mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1110a_Cases_Multishot_3--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1110b_Cases_Multishot_3--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1110c_Cases_Multishot_3--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1110d_Cases_Multishot_3--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train --batch_size=3    --env 1111-cases-multishot-3-fine-tuning         --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1111_retina50.log 2>&1
python main.py --purpose evaluate --env 1111-cases-multishot-3-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1111-cases-multishot-3-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1111 evals RetinaNet 50" && git push



# 1100, 1101
# ==========
python main.py --purpose train    --env 1100-cases    --start_from_xval_k=3                       --gpu_no 2 --batch_size=3 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ > train_1100_retina50.log 2>&1
python main.py --purpose evaluate --env 1100-cases                                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1100-cases-body-shots-eval                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1100 evals RetinaNet 50" && git push



mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50
mkdir -p /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50

cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100a_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100b_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100c_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50
cp /home/blsa/projects/confidential/wound-detection/checkpoints/1100d_Cases--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50

mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5
mv /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /home/blsa/projects/confidential/wound-detection/checkpoints/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train --batch_size=3    --env 1101-cases-fine-tuning                     --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/  > train_1101_retina50.log 2>&1
python main.py --purpose evaluate --env 1101-cases-fine-tuning                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1101-cases-fine-tuning-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

find /home/blsa/projects/confidential/wound-detection/evaluation -type f -name '*.csv' -exec cp {} /home/blsa/projects/ffhs-bachelor-thesis-wound-detection/evaluation/final/ \;
git add evaluation/final/* && git commit -m "Add 1101 evals RetinaNet 50" && git push
