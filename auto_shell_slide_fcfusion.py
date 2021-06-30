import subprocess

# bash scripts/train_fcfusion.sh valence stress deepspectrum 2 1

# target_list=['EDA', 'arousal', 'valence']
target_list=['EDA']
# task=$2
# feature_list=['egemaps', 'vggish', 'deepspectrum', 'deepspectrum,egemaps', 'vggish,egemaps']
feature_list=['egemaps', 'vggish', 'deepspectrum', 'deepspectrum,egemaps', 'vggish,egemaps', 'wav2vec', 'wav2vec,egemaps']
# feature_list=['vggface', 'denseface', 'vggface,denseface', 'bert', 'glove', 'bert,glove']
run_idx_list=['1', '2']
gpu_ids_list=['0', '0']
fusion_type = 'fcfusion'
loss_type = 'mse'
# lr_list = ['1e-4', '5e-4', '5e-5']
lr_list = ['1e-4', '5e-5']

# command = "echo a; echo b"
command = ""

for target in target_list:
    if (target == 'EDA'):
        task = "physio"
    else:
        task = "stress"
    
    for feature in feature_list:
        for run_gpu_index in range(2):
            for lr in lr_list:
                run_idx = run_idx_list[run_gpu_index]
                gpu_ids = gpu_ids_list[run_gpu_index]

                command += "bash scripts/train_slide_"+fusion_type+".sh "+target+' '+task+' '+feature+' '+run_idx+' '+gpu_ids+' '+loss_type+' '+lr+'; '


ret = subprocess.run(command, capture_output=True, shell=True)

# before Python 3.7:
# ret = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

print(ret.stdout.decode())