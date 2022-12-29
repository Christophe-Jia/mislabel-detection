#!/bin/sh

datadir=$1
dataset=$2
seed=$3
noise_ratio=$4
noise_type=$5
result_save_path=$6
detector_file=$7
remove_ratio=$8

net_type=resnet
depth=32

# td path 
td_file="${result_save_path}/${dataset}_${net_type}${depth}_percmislabeled${noise_ratio}_${noise_type}/computation4td_seed${seed}"

# Remove the identified mislabeled saples and retrain
savedir="${result_save_path}/${dataset}_${net_type}${depth}"
savedir="${savedir}_percmislabeled${noise_ratio}_${noise_type}/prune4retrain_seed${seed}/detector_${detector_files}/percremove${remove_ratio}"
args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${net_type} --depth ${depth}"
args="${args} --noise_ratio ${noise_ratio} --noise_type ${noise_type} --seed ${seed} --num_valid 0"
train_args="--num_epochs 300 --lr 0.1 --lr_drops 0.5, --wd 1e-4 --batch_size 256 --num_workers 4"
train_args="${train_args} --td_files ${td_file} --remove_ratio ${remove_ratio} --detector_files ${detector_file}"
              
cmd="python runner.py ${args} - train ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
