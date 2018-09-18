import logging
import datetime
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import CaptioningModel
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

# Device configuration for gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# save pytorch model
def save_checkpoint(state, args, filename='checkpoint.pth.tar'):
    checkpoint_name = str(args.model_path) + "/" + filename
    torch.save(state, checkpoint_name)

def validate(val_data, model, criterion):
    print("Validation started")
    logging.debug("Validation started")
    model.eval()
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_data):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, backward and optimize
            outputs = model(images, captions, lengths)  
            loss = criterion(outputs, targets)
    model.train()
    return loss

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader for train
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers)
    # 40K is too big for val set, select 5000.
    indices = np.random.choice(40000,5000)
    sampler = SubsetRandomSampler(indices)
    # Build data loader for val
    data_loader_val = get_loader(args.image_dir, args.caption_path_val, vocab,
                            transform, args.batch_size,
                            shuffle=False,
                            sampler=sampler, num_workers=args.num_workers)                             

    # Build the models
    model = CaptioningModel(args.embed_size, args.hidden_size, len(vocab),
                            args.num_layers).to(device)  
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    # by default starting epoch is 0
    starting_epoch = 0
    if args.input_model:
        checkpoint = torch.load(args.input_model)
        starting_epoch = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epochs {})"
              .format(args.input_model, checkpoint['epochs']))

    # Logging
    logging.debug("LR: {}, BatchSize: {}, HiddenSize: {},  Starting epoch:{},\
                ".format(args.learning_rate,
                        args.batch_size, 
                        args.hidden_size, 
                        starting_epoch))
    least_val_loss = float('inf')
    # Note that epoch indexing starts from zero.
    # Train the models
    total_step = len(data_loader)
    for epoch in tqdm(range(starting_epoch, args.num_epochs)):
        logging.debug('Starting epoch %d / %d \n' % (epoch, args.num_epochs))
        for i, (images, captions, lengths) in tqdm(enumerate(data_loader)):
            model.train()
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            # Forward, backward and optimize
            outputs = model(images, captions, lengths)  
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            log_result = ''
            if i % args.log_step == 0:
                # run validation data on model
                val_loss = validate(data_loader_val, model, criterion)
    
                if val_loss < least_val_loss:
                    val_loss = least_val_loss
                    # save model if val loss is improved
                    save_checkpoint({
                        'epochs': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, args, filename="CaptionModel_" + str(epoch) + ".pth.tar")
                
                log_result = ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, \
                             Val Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step,
                             loss.item(), np.exp(loss.item()), val_loss))
                logging.info(log_result)
                print(log_result)

       
      


### COMMAND LINE ARGUMENTS ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning Training')

    parser.add_argument('--model_path', type=str, default='models/',
                        help='path for saving trained models')
    parser.add_argument('--input_model', default='', type=str,
                        help='Path of model for loading')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--caption_path_val', type=str, default='data/annotations/captions_val2014.json',
                        help='path for val annotation json file')                        
    parser.add_argument('--log_step', type=int, default=200,
                        help='step size for prining log info')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in lstm')

    parser.add_argument('-e', '--num_epochs', type=int, default=25,
    help = 'total number of epoch')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)

    log_name = 'logs/'+ str(datetime.datetime.now()) + '.log'
    logging.basicConfig(
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_name),
            logging.StreamHandler()
        ],
        level=logging.DEBUG
    )
    main(args)