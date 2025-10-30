#!/bin/bash
#SBATCH --job-name=mon_test
#SBATCH --output=sortie.txt
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:30:00

~/miniconda3/envs/rub311/bin/python main.py --gpu cuda:0 config/model_tests/fracBNN/cifar10_fbnn_two_steps.toml