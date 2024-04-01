#!/bin/bash

##NESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Equidock_dropout_dips       #Set the job name to "JobExample4"
#SBATCH --time=60:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --mem=204800M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Equidock_dropout_dips.%1      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=132715540063             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=yxliu@tamu.edu    #Send all emails to email_address 

#First Executable Line
conda env list
source /home/yxliu/.bashrc
source activate protein-graph
cd ..
CUDA_VISIBLE_DEVICES=0 python -m src.train -dropout 0.05 -drop_message None -drop_message_rate 0 -drop_connect None -drop_connect_rate 0 -iegmn_n_lays 4 -patience 100