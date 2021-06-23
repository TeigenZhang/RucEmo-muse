set -e
target=$1
task=$2
feature=$3
run_idx=$4
gpu_ids=$5
loss_type=$6
lr=$7

cmd="python train.py --dataset_mode=muse_$task --model=fcfusion --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--max_seq_len=100 --regress_layers=128,128 --hidden_size=1024
--feature_set=$feature --target=$target
--batch_size=8 --lr=$lr --dropout_rate=0.3 --run_idx=$run_idx --verbose
--niter=30 --niter_decay=40 --num_threads=0 --loss_type=$loss_type
--name=baseline_fcfusion_wonorm --suffix={feature_set}_{target}_hidden{hidden_size}_seq{max_seq_len}_lr{lr}_{loss_type}loss_run{run_idx}"

# echo "\n-----------------------------------------------------------------------------------"
# echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
