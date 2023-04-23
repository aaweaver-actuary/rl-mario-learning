@echo off
@REM conda create -p ./envs/mario python=3.10 ipykernel jupyter notebook pytorch torchvision numpy pandas=2 gym=0.26 plotly scipy pillow pyvirtualdisplay imageio -c pytorch -c conda-forge
conda env create -f mario_env.yml -n mario-env
