#!/bin/bash

python main.py --purpose train    --env 1100-cases                                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1100-cases                                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1100-cases-body-shots-eval                                --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

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

python main.py --purpose train    --env 1101-cases-fine-tuning                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1101-cases-fine-tuning                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1101-cases-fine-tuning-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval


python main.py --purpose train    --env 1110-cases-multishot-3                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1110-cases-multishot-3                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1110-cases-multishot-3-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

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

python main.py --purpose train    --env 1111-cases-multishot-3-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1111-cases-multishot-3-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1111-cases-multishot-3-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval


python main.py --purpose train    --env 1112-cases-multishot-6                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1112-cases-multishot-6                                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1112-cases-multishot-6-body-shots-eval                    --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

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

python main.py --purpose train    --env 1113-cases-multishot-6-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1113-cases-multishot-6-fine-tuning                        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/
python main.py --purpose evaluate --env 1113-cases-multishot-6-fine-tuning-body-shots-eval        --gpu_no 2 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/home/blsa/projects/confidential/wound-detection/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval

cp /home/blsa/projects/confidential/wound-detection/evaluation/1100a_Cases--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1100b_Cases--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1100c_Cases--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1100d_Cases--RetinaNet-Resnet50/*.csv ./evaluation/final

cp /home/blsa/projects/confidential/wound-detection/evaluation/1101a_Cases_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1101b_Cases_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1101c_Cases_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1101d_Cases_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final

cp /home/blsa/projects/confidential/wound-detection/evaluation/1110a_Cases_Multishot_3--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1110b_Cases_Multishot_3--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1110c_Cases_Multishot_3--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1110d_Cases_Multishot_3--RetinaNet-Resnet50/*.csv ./evaluation/final

cp /home/blsa/projects/confidential/wound-detection/evaluation/1111a_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1111b_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1111c_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1111d_Cases_Multishot_3_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final

cp /home/blsa/projects/confidential/wound-detection/evaluation/1112a_Cases_Multishot_6--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1112b_Cases_Multishot_6--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1112c_Cases_Multishot_6--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1112d_Cases_Multishot_6--RetinaNet-Resnet50/*.csv ./evaluation/final

cp /home/blsa/projects/confidential/wound-detection/evaluation/1113a_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1113b_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1113c_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final
cp /home/blsa/projects/confidential/wound-detection/evaluation/1113d_Cases_Multishot_6_Fine_Tuning--RetinaNet-Resnet50/*.csv ./evaluation/final

git add evaluation/final/* && git commit -m "Add 11x evals RetinaNet 50" && git push