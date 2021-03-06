"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \
"""

##
# LIBRARIES

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model
import torch

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

# WandB – Import the wandb library
import wandb

##
def main():
    """ Training
    """
    torch.autograd.set_detect_anomaly(True)
    wandb.init(entity="wenxun", project="tutorial")
    opt = Options().parse()
    data = load_data(opt)
    model = load_model(opt, data)
    model.train()

if __name__ == '__main__':
    main()
