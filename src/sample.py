import torch
#import matplotlib.pyplot as plt
import numpy as np 
import argparse
import os
from torchvision import transforms 
from src.model import CaptioningModel
from PIL import Image
import json


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    # useful if png image with 4 channel is uploaded
    image = image.convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)   
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary JSON
    try:
        voc = json.load(open(args['vocab_path'], 'rb'))
    except:
        raise IOError("Please download vocab.json by using the script\
                     at src/vocab/download.sh")

    # Build the model
    model = CaptioningModel(args['embed_size'], args['hidden_size'], voc['length'],
                            args['num_layers']).to(device)  
    # Load pretrained model
    checkpoint = torch.load(args['model_path'])
    model.load_state_dict(checkpoint['state_dict'])
    # Switch to eval mode, this is necessary for dropout, batchnorm, etc since
    # they behave differently in evaluation mode
    model.eval()    
    # Transfer model to gpu or stay in cpu
    model.to(device)
    

    # Prepare an image
    image = load_image(args['image'], transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    sampled_ids = model.sample(image_tensor)
    sampled_ids = sampled_ids[0].cpu().numpy()
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = voc['idx2word'][str(word_id)]
        if word != '<start>' and word != '<end>':
            sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    return sentence
    
    # Print out the image and the generated caption
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('-m', '--model_path', type=str, default='src/model/deploy_model.pth.tar', help='path for trained model')
    parser.add_argument('--vocab_path', type=str, default='src/vocab/vocab.json', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
