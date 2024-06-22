#!/bin/bash

#PBS -u gaurav
#PBS -N GAURAV
#PBS -q gpu
#PBS -l select=1:ncpus=40:ngpus=2
#PBS -o out.log
#PBS -j oe

cd /home/sonali/gaurav

#module load utils/anaconda3.5

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cat $PBS_NODEFILE | uniq > nodes
nvidia-smi | tee nv.log

python3 hello.py

