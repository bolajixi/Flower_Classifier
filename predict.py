import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np

from PIL import Image
import time
import argparse
from utils import load_checkpoint, load_cat_to_name, parse_gpu_arg

def parse_args():
    parser = argparse.ArgumentParser(description='Predict image classification')

    parser.add_argument('--img_path', type=str, help='Path to image prediction')
    parser.add_argument('--checkpoint', help='Model checkpoint to use for prediction', default='ImageClassifier/checkpoint.pth')
    parser.add_argument('--top_k', help='Number of top predictions', type=int, default=5)
    parser.add_argument('--category_names', type=str, help='File containing category names', default='ImageClassifier/cat_to_name.json')
    parser.add_argument('--gpu', type=str, help='Make use of GPU if available (true|false), (yes|no), (y|n), (1|0)')

    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    size_256 = 256
    crop_size = 224

    # Augment data using PyTorch Transfroms
    img_pil = Image.open(image, mode='r')

    img_trans = transforms.Compose([transforms.Resize(size_256),
                                    transforms.CenterCrop(crop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ])
    img_processed = img_trans(img_pil)

    return img_processed

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    img = process_image(image_path)  # Process image before making predictions
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()
    model.eval()

    with torch.no_grad():
        output = model.forward(img.to(device))

        ps = torch.exp(output).data
        top_p, top_class = ps.topk(topk)

    model.train()
    return top_p, top_class

def main():
    args = parse_args()
    if args.img_path:
        img_path = args.img_path
    else:
        raise argparse.ArgumentError(args.img_path, 'File image path required')
        
    top_probs = args.top_k
    
    gpu = parse_gpu_arg(args.gpu)

    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_to_name(args.category_names)
    
    start_time = time.time()
    probs, classes = predict(img_path, model, args.top_k, gpu)
    labels = [cat_to_name[str(idx+1)] for idx in np.array(classes[0])]
    probability =  np.array(probs[0])

    i=0
    print('{:*^50s}\n'.format('Prediction Results'))
    while i < top_probs:
        print('Image: {} \t{:>15}: {:.4f}%'.format(labels[i], 'Probability', probability[i]))
        i += 1

    total_prediction_time = time.time() - start_time

    print('\n{:*^50s}'.format(''))
    print('Prediction complete...')
    print('Total Prediction time: {:.3f}s'.format(total_prediction_time % 60))
    
    
    
if __name__ == "__main__":
    main()