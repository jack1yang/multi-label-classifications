import torch 
import torch.nn as nn
from torch.autograd import Variable 
from torchvision import transforms
from dataset_chexpert import *
import math
import os 
import argparse
from model import EncoderCNN_singleimage_frontal, EncoderCNN_singleimage_lateral
from mAP import *


sigmoid = nn.Sigmoid()
# Variable wrapper
def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )

def main(args):
    # To reproduce training results
    torch.manual_seed( args.seed )
    if torch.cuda.is_available():
        torch.cuda.manual_seed( args.seed )
        
    # Create model directory
    if not os.path.exists( args.model_path ):
        os.makedirs(args.model_path)
    
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

    model = EncoderCNN_singleimage_frontal(args.num_classes)
   


    if args.pretrained:
        model_pretrained = torch.load( args.pretrained )
        model.load_state_dict( model_pretrained, strict=False )
        # Get starting epoch #, note that model is named as '...your path to model/algoname-epoch#.pkl'
        # A little messy here.
        start_epoch = int( args.pretrained.split('/')[-1].split('-')[1].split('.')[0] ) + 1
        
    else:
        start_epoch = 1


    # mult_label loss
    #bce_loss = nn.BCELoss()
    MIL_loss = nn.MultiLabelSoftMarginLoss()

    # changed to  GPU mode if available
    if torch.cuda.is_available():
        model.cuda()
        #bce_loss.cuda()
        MIL_loss.cuda()

    #dp_model = torch.nn.DataParallel(model, device_ids=[0,1])

    # Train the model 
    total_step = len(data_loader)
    print('len(data_loader)',total_step)

    learning_rate = args.learning_rate
    # starting training
    max_ap = 0
    for epoch in range(start_epoch, args.num_epochs + 1):
        # start decay learning rate
        if epoch > args.lr_decay:

            frac = (epoch - args.lr_decay) / args.learning_rate_decay_every
            decay_rator = math.pow(0.8, frac)

            # decay the learning rate 
            learning_rate = args.learning_rate * decay_rator

        print('Learning rate for Epoch %d: %.6f'%(epoch, learning_rate))


        # Constructing model parameters for optimization
        optimizer = torch.optim.Adam( model.parameters(), lr = learning_rate,
                                betas = (args.alpha, args.beta))

        print( '-------------Train for Epoch %d --------------'% epoch)
        for i, (images, image_name, image_labels) in enumerate(data_loader):
            
            images = to_var(images)
            image_labels = torch.FloatTensor(image_labels)
            image_labels = to_var(image_labels)

            model.train()
            #dp_model.train()
            optimizer.zero_grad()

            outputs = model(images)

            loss = MIL_loss(outputs, image_labels)
            loss.backward()

            optimizer.step()

  
      

            # Print log info
            if i % args.log_step == 0:
                print ('Epoch [%d/%d], Step [%d/%d], BCE Loss: %.4f'%( epoch, 
                                                                                                 args.num_epochs, 
                                                                                                 i, total_step, 
                                                                                                 loss.item()))
            


        gts =[]
        outputs = []
        with torch.no_grad():
            for i, (images, image_name, image_labels) in enumerate(data_loader):
                
                images = to_var(images)
                
                model.eval()
                #dp_model.train()

                output = model(images)
                output = sigmoid(output)
                output = output.data.cpu().numpy()

                gts.append(image_labels)
                outputs.append(output)
                if i == 96:
                    break
            outputs = np.stack(outputs, 1).reshape(-1, 82)
            gts = np.stack(gts, 1).reshape(-1, 82)
            # outputs = outputs.data.cpu().numpy()
            # image_labels = image_labels.data.cpu().numpy()
            outs=[]
            for cls in range(82):
                #print(roc_auc_score( gts[:,cls],predictions[:,cls]))
                ap = voc_eval(cls, outputs, gts)
                outs.append(ap)
            print(np.mean(outs))
            print(outs)
            # if np.mean(outs) > 0.13:
            #     np.savetxt(args.predictions, outputs)
            #     np.savetxt(args.gts, gts)
            #     return 
            # if epoch % 1 == 0:    
            if np.mean(outs) > max_ap:    
                max_ap = np.mean(outs)        
                # Save the  model after each epoch
                torch.save( model.state_dict(), 
                            os.path.join( args.model_path, 
                            'Resnet_single_image-%d.pkl'%( epoch ) ) ) 
            #     return 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( '-f', default='self', help='To make it runnable in jupyter' )

    parser.add_argument( '--model_path', type=str, default='frontal_pretrained_224/',
                         help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    
    parser.add_argument('--image_dir', type=str, default='dataset/NLMCXR_png_pairs' ,
                        help='directory for resized training images')
   
    parser.add_argument('--num_classes', type=int, default=82, help="classes number" )
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for printing log info')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed for model reproduction')
    parser.add_argument('--file_list', type=str, default='frontal_train_data.txt',
                        help='file which store the labels data')
   
    
    # ---------------------------Hyper Parameter Setup------------------------------------
    

    
    # Optimizer Adam parameter
    parser.add_argument( '--alpha', type=float, default=0.8,
                         help='alpha in Adam' )
    parser.add_argument( '--beta', type=float, default=0.999,
                         help='beta in Adam' )
    parser.add_argument( '--learning_rate', type=float, default=1e-4,
                         help='learning rate for the whole model' )

    
    
    # Training details          
    parser.add_argument( '--pretrained', type=str, default='image-35.pkl', help='start from checkpoint or scratch' )
    parser.add_argument( '--num_epochs', type=int, default=250 )
    parser.add_argument( '--batch_size', type=int, default=32 ) # on cluster setup, 60 each x 4 for Huckle server
    

    parser.add_argument( '--num_woeval_sizerkers', type=int, default=4 )
    parser.add_argument( '--clip', type=float, default=0.1 )
    parser.add_argument( '--lr_decay', type=int, default=40, help='epoch at which to start lr decay' )
    parser.add_argument( '--learning_rate_decay_every', type=int, default=10,
                         help='decay learning rate at every this number')
    
    args = parser.parse_args()
    
    print ('------------------------Model and Training Details--------------------------')
    print(args)
    
    # Start training
    main( args )
