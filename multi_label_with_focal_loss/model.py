import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from dataset_chexpert import *

class EncoderCNN_two_images( nn.Module ):
    def __init__( self, num_classes):
        super( EncoderCNN, self ).__init__()

        # ResNet-152 backend
        resnet_frontal = models.resnet152()
        modules_frontal = list( resnet_frontal.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv_frontal = nn.Sequential( *modules_frontal ) # last conv feature
        self.avgpool_frontal = nn.AvgPool2d( 10 )
        self.resnet_conv_frontal = resnet_conv_frontal
        self.affine_VI_frontal = nn.Linear( 2048, num_classes ) # reduce the dimension

        resnet_lateral = models.resnet152()
        modules_lateral =list( resnet_lateral.children() )[:-2]
        resnet_conv_lateral = nn.Sequential( *modules_lateral)
        self.avgpool_lateral = nn.AvgPool2d( 10 )
        self.resnet_conv_lateral = resnet_conv_lateral
        self.affine_VI_lateral = nn.Linear( 2048, num_classes)
        # Dropout before affine transformation
        self.dropout = nn.Dropout( 0.5 )

        self.init_weights()

    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_VI_frontal.weight, mode='fan_in' )
        self.affine_VI_frontal.bias.data.fill_( 0 )
        init.kaiming_uniform( self.affine_VI_lateral.weight, mode='fan_in' )
        self.affine_VI_lateral.bias.data.fill_( 0 )

    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        image_frontal = images[0]
        image_lateral = images[1]
        # Last conv layer feature map
        A_frontal = self.resnet_conv_frontal( image_frontal )
        A_lateral = self.resnet_conv_lateral( image_lateral ) #shape = (btach_size, 2048, 7, 7)

        avg_frontal = self.avgpool_frontal( A_frontal )
        avg_frontal = avg_frontal.view( avg_frontal.size(0), -1)
        avg_frontal = self.affine_VI_frontal( avg_frontal )

        # share the models
        # A_lateral = self.resnet_conv_frontal( image_lateral )
        # avg_lateral = self.avgpool_frontal( A_lateral )
        # avg_lateral = avg_lateral.view( avg_lateral.size(0), -1)
        # avg_lateral = self.affine_VI_frontal( avg_lateral )

        avg_lateral = self.avgpool_lateral( A_lateral )
        avg_lateral = avg_lateral.view( avg_lateral.size(0), -1)
        avg_lateral = self.affine_VI_lateral( avg_lateral)
        # V = [ v_1, v_2, ..., v_49 ]
        #V_frontal = A_frontal.view( A_frontal.size( 0 ), A.size( 1 ), -1 ).transpose( 1,2 )
        #V = F.relu( self.affine_VI( self.dropout( V ) ) )
        outputs = avg_frontal, avg_lateral
        return outputs

class EncoderCNN_singleimage_frontal(nn.Module):
    def __init__(self, num_classes):
        super(EncoderCNN_singleimage_frontal, self).__init__()


        # ResNet-152 backend
        resnet = models.resnet152()
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        self.avgpool = nn.AvgPool2d( 7 )
        self.resnet_conv_frontal = resnet_conv
        self.affine_VI_frontal_82 = nn.Linear( 2048, num_classes ) # reduce the dimension
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_VI_frontal_82.weight, mode='fan_in' )
        self.affine_VI_frontal_82.bias.data.fill_( 0 )
        

    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        
        # Last conv layer feature map
        images = self.resnet_conv_frontal( images )
        
        avg = self.avgpool( images )
        avg = avg.view( avg.size(0), -1)
        avg = self.affine_VI_frontal_82( avg )
        pred = self.sigmoid(avg)
        
        return pred

class EncoderCNN_singleimage_lateral(nn.Module):
    def __init__(self, num_classes):
        super(EncoderCNN_singleimage_lateral, self).__init__()


        # ResNet-152 backend
        resnet = models.resnet152()
        modules = list( resnet.children() )[ :-2 ] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential( *modules ) # last conv feature
        self.avgpool = nn.AvgPool2d( 7 )
        self.resnet_conv_lateral = resnet_conv
        self.affine_VI_lateral_82 = nn.Linear( 2048, num_classes ) # reduce the dimension
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights( self ):
        """Initialize the weights."""
        init.kaiming_uniform( self.affine_VI_lateral_82.weight, mode='fan_in' )
        self.affine_VI_lateral_82.bias.data.fill_( 0 )
        

    def forward( self, images ):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        
        # Last conv layer feature map
        images = self.resnet_conv_lateral( images )
        
        avg = self.avgpool( images )
        avg = avg.view( avg.size(0), -1)
        avg = self.affine_VI_lateral_82( avg )
        pred = self.sigmoid(avg)
        
        return pred


if __name__ == '__main__':

    # set same initial weights
    init_seed = 1
    torch.manual_seed(init_seed)

    net = EncoderCNN_singleimage()
    image_frontal = torch.randn(1,3,224,224)
    image_lateral = torch.randn(1,3,224,224)
    images=[image_frontal,image_lateral]
    ouputs = net(images)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    optimizer.zero_grad()
    target_frontal =torch.empty(1,dtype=torch.long).random_(4)
    target_lateral = torch.empty(1, dtype=torch.long).random_(4)
    loss_frontal = loss(ouputs[0], target_frontal)
    loss_lateral = loss(ouputs[1], target_lateral)
    total_loss = loss_frontal + loss_lateral
    total_loss.backward()
    #loss_frontal.backward()
    optimizer.step()
    #optimizer.zero_grad()
    #loss_lateral.backward()
    #optimizer.step()
    a=10
