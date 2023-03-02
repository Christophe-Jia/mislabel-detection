#!/bin/sh

datadir=$1
dataset=$2
seed=$3
detector_file=$4
remove_ratio=$5

result_save_path=replication
net_type=resnet
depth=50

# td path 
td_file="${result_save_path}/${dataset}_${net_type}${depth}/computation4td_seed${seed}"

# Remove the identified mislabeled samples and retrain
savedir="${result_save_path}/${dataset}_${net_type}${depth}"
savedir="${savedir}/prune4retrain_seed${seed}/detector_${detector_files}/percremove${remove_ratio}"
args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${net_type} --depth ${depth}"
args="${args} --seed ${seed} --num_valid 0"
train_args="--num_epochs 300 --lr 0.1 --lr_drops 0.33,0.67, --wd 1e-4 --batch_size 256 --num_workers 4"
train_args="${train_args} --td_files ${td_file} --remove_ratio ${remove_ratio} --detector_files ${detector_file}"
              
cmd="python runner.py ${args} - train ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
