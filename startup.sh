echo "Create conda environment with jax"
yes | conda create -n env-proj jupyter pip
eval "$(conda shell.bash hook)"
conda activate env-proj
pip3 install --upgrade pip
pip3 install --upgrade "jax[cpu]"
pip3 install --upgrade ipympl
pip3 install --upgrade matplotlib
pip install -r requirements.txt
