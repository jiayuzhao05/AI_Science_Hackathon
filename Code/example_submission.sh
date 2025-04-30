#!/usr/bin/bash
#SBATCH --account=ai4s-hackathon
#SBATCH --reservation=ai4s-hackathon
#SBATCH -p schmidt-gpu
#SBATCH --qos=schmidt
#SBATCH --gres=gpu:1
#SBATCH --time 10:00


module load python/miniforge-24.1.2 # python 3.10

echo "output of the visible GPU environment"
nvidia-smi

# Use material characterization environment
source activate /project/ai4s-hackathon/ai-sci-hackathon-2025/envs/gnnpytorch

echo PyTorch
python example_torch.py

conda deactivate

# Use rl and biological networks environment
source activate /project/ai4s-hackathon/ai-sci-hackathon-2025/envs/rl+bnpytorch

echo PyTorch
python example_torch.py

# Use a different environment for JAX
#source /project/dfreedman/hackathon/hackathon-env-jax/bin/activate
#echo JAX
#python example_jax.py
