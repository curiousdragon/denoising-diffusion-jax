echo "create conda environment with jax"
yes | conda create -n env-proj
conda activate env-proj
pip3 install --upgrade pip
pip3 install --upgrade "jax[cpu]"
