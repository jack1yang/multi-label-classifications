import torch 
import torch.nn as nn
from torch.autograd import Variable 
from torchvision import transforms
from dataset_chexpert import *
import math
import os 
import argparse
from model import EncoderCNN_singleimage_frontal, EncoderCNN_singleimage_lateral, EncoderCNN_two_images

# variable warpper
def to_var(x, volatile=False):
    """
    Warpper torch tensor into variable
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile = volatile)


def main(argas):
    # to reproduce testing results
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Image Preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop( args.crop_size ),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                             ( 0.229, 0.224, 0.225 ))])
    
    data_loader = get_loader(args.image_dir, args.file_list, 
                            transform, args.batch_size, True)

    #model = EncoderCNN_singleimage_frontal(args.num_classes)
    model = EncoderCNN_singleimage_frontal(args.num_classes)

    
    if args.pretrained:
        
        model.load_state_dict( torch.load( args.pretrained ) )
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int( args.pretrained.split('/')[-1].split('-')[1].split('.')[0] ) + 1
        
    else:
        start_epoch = 1


    # changed to  GPU mode if available
    if torch.cuda.is_available():
        model.cuda()
        
    outputs = {}
    gts = {}
    print('len(data_loader): %d'%len(data_loader))
 
    with torch.no_grad():
        for i, (images, image_name, image_labels) in enumerate(data_loader):
            
            images = to_var(images)
            
            model.eval()
            #dp_model.train()

            output = model(images)
            output = output.data.cpu().numpy()

            image_label = image_labels[0].tolist()
            gts[image_name[0]] = image_label
            
            output = output[0].tolist()
            outputs[image_name[0]] = output
                 
            print(i)

            # gts.append(image_labels)
            # outputs.append(output)

    with  open(args.predictions, 'w')  as f:
        json.dump(outputs, f)

    with open(args.gts, 'w') as f:
        json.dump(gts, f)
    # predict_write =  open(args.predictions, 'w') 
    # write_f.write(output + '\n')
    # gts_write =  open(args.gts, 'w') 
    # write_f.write(image_label, + '\n')
    
    # outputs = np.stack(outputs, 1).squeeze()
    # gts = np.stack(gts, 1).squeeze()
    # np.savetxt(args.predictions, outputs)

    # np.savetxt(args.gts, gts)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, default='predictions_lateral_train224.json')
    parser.add_argument('--gts', type=str, default='gts_lateral_train224.json')
    parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )

    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    
    parser.add_argument('--image_dir', type=str, default='dataset/NLMCXR_png_pairs' ,
                        help='directory for resized training images')
   
    parser.add_argument('--num_classes', type=int, default=82, help="classes number" )

    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    parser.add_argument('--file_list', type=str, default='lateral_train_data.txt',
                        help='file which store the labels data')
   
    
    
    # Training details
    parser.add_argument( '--pretrained', type=str, default='/media/mdisk/ysk/code/medical_long_sentence/test_concepts/lateral_pretrained_224/Resnet_single_image-206.pkl', help='start from checkpoint or scratch' )
    parser.add_argument( '--batch_size', type=int, default=1 ) # on cluster setup, 60 each x 4 for Huckle server
    

    
    args = parser.parse_args()
    
    print ('------------------------Model and Testing Details--------------------------')
    print(args)
    
    # Start training
    main( args )
