srun:
	srun --gres=gpu:L40S:1 --mincpus=16 --pty bash

srun-mnist:
	srun --gres=gpu:L40S:1 --mincpus=16 --time=01:00:00 --pty bash

srun-cifar10:
	srun --gres=gpu:L40S:1 --mincpus=16 --time=02:00:00 --pty bash

srun-cifar100: # Temps à modifier
	srun --gres=gpu:L40S:1 --mincpus=16 --time=24:00:00 --pty bash

srun-imagenet: # Temps et GPU à modifier
	srun --gres=gpu:L40S:3 --mincpus=16 --time=24:00:00 --pty bash