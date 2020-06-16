import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse
import json

def load_cat_to_name(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def parse_gpu_arg(v):
    '''
        This function converts commandline argument to boolean
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for --gpu.')

def save_checkpoint(path, model, optimizer, classifier, args):
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'epochs': args.epochs,
                  'classifier' : classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
    print('Saving checkpoint....')
    torch.save(checkpoint, path)
    print('Checkpoint saved')
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model