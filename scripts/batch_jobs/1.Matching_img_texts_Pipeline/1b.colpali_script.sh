#!/bin/bash
#SBATCH --job-name=colipali
#SBATCH --partition=publicgpu   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=32G               
#SBATCH --gres=gpu:1    
#SBATCH --time=02:00:00         
#SBATCH --output=/home2020/home/icube/janil/Rhine_proj/logs/colpali_%j.log

export RHINE_PROJECT_ROOT="${RHINE_PROJECT_ROOT:-/home2020/home/icube/janil/Rhine_proj}"


# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpali-v1.2', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/colpali-v1.2', '--token', 'YOUR_HF_TOKEN'], env=env)"

# python3 -c "import os, subprocess; env = os.environ.copy(); env.update({'HF_HUB_DISABLE_SYMLINKS': '1', 'HF_HOME': '/tmp/hf_temp'}); subprocess.run(['hf', 'download', 'vidore/colpaligemma-3b-pt-448-base', '--local-dir', '$RHINE_PROJECT_ROOT/hf_models/colpaligemma-3b-pt-448-base', '--token', 'YOUR_HF_TOKEN'], env=env)"

{
    echo "--- CHECKING GPU ALLOCATION ---"
    nvidia-smi

    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate colpali_env  
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    # echo "Starting ColiPali extraction..."
    # python -u ~/Rhine_proj/scripts/code/1.Matching_img_texts_Pipeline/2.colipali_exp.py

    echo "Starting ColiPali extraction with weights..."
    python -u ~/Rhine_proj/scripts/code/1.Matching_img_texts_Pipeline/2a.colpali_exp_with_weights.py

} 2>&1 | tee ~/Rhine_proj/outputs/live_logs/1.Image_matching_output/colpali_weights_output.txt