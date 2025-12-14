#!/usr/bin/env bash
conda env create --name=metrabs_pt --file=environment.yml -y
conda activate metrabs_pt
pip3 install torch torchvision torchaudio
pip install hydra-core
pip install ultralytics
pip install pyav
#### PosePile is needed but the installation file is broken. There is a fix in the patch-1 branch of kelpabc123 but it doesn't seem to work with pip install from that branch
pip install git+https://github.com/Vittorio-Caggiano/PosePile.git
wget -O - https://bit.ly/metrabs_l_pt | tar -xzvf -
### on macos
## curl -L https://bit.ly/metrabs_l_pt | tar -xzvf -
export DATA_ROOT="" # Most of the code expects DATA_ROOT to exist, even if empty. see https://github.com/isarandi/metrabs/issues/72#issuecomment-1778991215
