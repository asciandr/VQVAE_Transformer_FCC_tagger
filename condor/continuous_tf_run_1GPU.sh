#!/bin/bash

#cp -r /usatlas/u/asciandra/tokenization/code/ .
cd /usatlas/u/asciandra/tokenization/
singularity shell -B /gpfs01/ --nv /usatlas/u/asciandra/colorsinglet.sif <<EOF
#export https_proxy=http://proxy.sdcc.bnl.local:3128/
#export OMP_NUM_THREADS=4
#export NCCL_DEBUG=INFO
#export NCCL_P2P_LEVEL=NVL

cd code/
python3 tf_continuous_training.py
#cp *.pt /usatlas/u/asciandra/tokenization/code/

#wandb login
#torchrun --standalone --nnodes=1 --nproc_per_node=1 -m weaver.train --data-train /atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_*.root --data-config example_7labels.yaml --network-config example_ParticleTransformer.py --model-prefix TRAINING_highStatsBaseline_7labels_1GPU --num-workers 0 --gpus 0 --batch-size 2048 --start-lr 1e-3 --num-epochs 40 --optimizer ranger --fetch-step 0.01 --backend nccl --log-wandb --wandb-displayname highStatsBaseline_7labels_1GPU_2 --wandb-projectname first_BNL_GPU_test_1GPU

EOF

