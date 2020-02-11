#!/bin/bash

#python main.py --purpose train    --env 1030-joint-puppet-close-up-wounds                                       --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ > train-1030-retina50.log 2>&1
#python main.py --purpose evaluate --env 1030-joint-puppet-close-up-wounds                                       --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/                                     --eval_images --eval_heatmaps
#python main.py --purpose evaluate --env 1030-joint-puppet-close-up-wounds-body-shots-eval                       --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-body-shots-eval --eval_images --eval_heatmaps
#python main.py --purpose evaluate --env 1030-joint-puppet-close-up-wounds-cases-eval                            --gpu_no 0 --net_type retina-resnet50 --data_dir=/home/blsa/projects/confidential/wound-detection/data/ --checkpoint_dir=/checkpoints/ --eval_dir=/home/blsa/projects/confidential/wound-detection/evaluation/ --eval_name_suffix=-cases-eval      --eval_images --eval_heatmaps

mkdir -p /checkpoints/1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /checkpoints/1030a_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mv /checkpoints/1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1031a_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5

mkdir -p /checkpoints/1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /checkpoints/1030_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mv /checkpoints/1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1031b_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5

mkdir -p /checkpoints/1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /checkpoints/1030c_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mv /checkpoints/1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1031c_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5

mkdir -p /checkpoints/1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
cp /checkpoints/1030d_Joint_Puppet_Closeup_Wounds--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50
mv /checkpoints/1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/*latest.h5 /checkpoints/1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50/1031d_Joint_Puppet_Closeup_Wounds_Fine_Tuning--RetinaNet-Resnet50_0001.h5

python main.py --purpose train    --env 1031-joint-puppet-close-up-wounds-fine-tuning                           --gpu_no 3 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation > train-1031-retina50.log 2>&1
python main.py --purpose evaluate --env 1031-joint-puppet-close-up-wounds-fine-tuning                           --gpu_no 3 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1031-joint-puppet-close-up-wounds-fine-tuning-body-shots-eval           --gpu_no 3 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps
python main.py --purpose evaluate --env 1031-joint-puppet-close-up-wounds-fine-tuning-cases-eval                --gpu_no 3 --net_type retina-resnet50 --data_dir=/home/samuelblattner/projects/ffhs-bachelor-thesis-wound-detection/data/vanilla_datasets --checkpoint_dir=/checkpoints --eval_dir=/checkpoints/evaluation                                     --eval_images --eval_heatmaps