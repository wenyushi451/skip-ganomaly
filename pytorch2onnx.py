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
from lib.models import load_model, load_then_export_model
import torch

# Ignore excessive warnings
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)

##
def main():
    """ Training
    """
    opt = Options().parse()
    data = load_data(opt)
    load_then_export_model(opt, data)

if __name__ == '__main__':
    main()
