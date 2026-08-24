#!/bin/bash
#SBATCH --job-name=BLIP_Captioning
#SBATCH --partition=public       
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      
#SBATCH --mem=32G                 # 16GB RAM is plenty for BLIP
#SBATCH --time=01:00:00          
#SBATCH --output=blip_job_%j.out  # Save terminal output to this file

# 1. Activate Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate vlm_env            

# Run the script
python -u ~/Rhine_proj/scripts/code/2.Image_description_Pipeline/0.MFlorence_oneshot_descp.py > ~/Rhine_proj/outputs/live_logs/florence_output.txt 2>&1