#!/bin/sh

datadir=$1
dataset=$2
seed=$3
noise_ratio=$4
noise_type=$5
net_type=$6
depth=$7
result_save_path=replication

# General arguments for training dynamics computation
args="--data ${datadir}/${dataset} --dataset ${dataset} --net_type ${net_type} --depth ${depth}"
args="${args} --noise_ratio ${noise_ratio} --noise_type ${noise_type} --seed ${seed} --num_valid 0"
train_args="--num_epochs 200 --lr 0.1 --wd 1e-4 --batch_size 128 --num_workers 4"

if [ "$dataset" = "webvision50" ] || [ "$dataset" = "clothing100k" ];then
    savedir="${result_save_path}/${dataset}_${net_type}${depth}/computation4td_seed${seed}"
else
    savedir="${result_save_path}/${dataset}_${net_type}${depth}_percmislabeled${noise_ratio}_${noise_type}/computation4td_seed${seed}"
fi

cmd="python runner.py ${args} --save ${savedir}  - train_for_td_computation ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
