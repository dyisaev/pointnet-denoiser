import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_utils import BatchNorm1D_controlled_bias, count_parameters, count_parameters_requiring_grad

class PointNet_Encoder(nn.Module):
    """
    PointNet Encoder class for PointNet Denoiser  
    Input: 3D point cloud (B x POINT_DIM x N )   
    """
    def __init__(self,point_dim=3, output_dim = 1024, use_bias=True):
        super(PointNet_Encoder, self).__init__()
        self.output_dim=output_dim
        self.output_shape=(output_dim,1)

        self.conv1 = nn.Conv1d(point_dim, 64, 1, bias=use_bias)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=use_bias)
        self.conv3 = nn.Conv1d(64, 64, 1, bias=use_bias)
        self.conv4 = nn.Conv1d(64, 128, 1, bias=use_bias)
        self.conv5 = nn.Conv1d(128, output_dim, 1, bias=use_bias)
        
        #  Batch norm with potentially no bias
        self.bn1 = BatchNorm1D_controlled_bias(64,bias=use_bias)
        self.bn2 = BatchNorm1D_controlled_bias(64,bias=use_bias)
        self.bn3 = BatchNorm1D_controlled_bias(64,bias=use_bias)
        self.bn4 = BatchNorm1D_controlled_bias(128,bias=use_bias)
        self.bn5 = BatchNorm1D_controlled_bias(output_dim,bias=use_bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))


        global_feat = nn.MaxPool1d(x.size(-1))(x)
        print('global feat shape',global_feat.shape)  # torch.Size([32, 1024, 1])
        return global_feat
class PointNet_Decoder(nn.Module):
    """
    PointNet Decoder class for PointNet Denoiser
    Input: 3D point cloud (B x input_dim x N )
    Output: 3D point cloud (B x point_dim x N )
    """
    def __init__(self,input_dim, point_dim=3, use_bias=True):
        super(PointNet_Decoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 512, 1, bias=use_bias)
        self.conv2 = nn.Conv1d(512, 256, 1, bias=use_bias)
        self.conv3 = nn.Conv1d(256, 128, 1, bias=use_bias)
        self.conv4 = nn.Conv1d(128, 64, 1, bias=use_bias)   
        self.conv5 = nn.Conv1d(64, point_dim, 1, bias=use_bias)

        #  Batch norm with potentially no bias
        self.bn1 = BatchNorm1D_controlled_bias(512,bias=use_bias)
        self.bn2 = BatchNorm1D_controlled_bias(256,bias=use_bias)
        self.bn3 = BatchNorm1D_controlled_bias(128,bias=use_bias)
        self.bn4 = BatchNorm1D_controlled_bias(64,bias=use_bias)
        self.bn5 = BatchNorm1D_controlled_bias(point_dim,bias=use_bias)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        return x
        
class PointNet_Concatenator(nn.Module):
    """
    PointNet Concatenator class for PointNet Denoiser
    Input: 3D point cloud (B x point_dim x N ), shape decriptor (Bx1024x1)
    """
    def __init__(self):
        super(PointNet_Concatenator, self).__init__()
    def forward(self, x, y):
        y = y.expand(-1,-1,x.shape[2])
        x = torch.cat((x,y),1)
        return x
class PointNet_Denoiser(nn.Module):
    """
    PointNet Denoiser class
    Input: 3D point cloud (B x point_dim x N )
    Output: 3D point cloud (B x point_dim x N )  
    POINT_DIM may contain original point cloud x,y,z coords and encoded representation concatenated to it   
    """
    def __init__(self, point_dim=3, use_bias=True):
        super(PointNet_Denoiser, self).__init__()
        self.encoder = PointNet_Encoder(point_dim=point_dim, use_bias=use_bias)
        self.concatenator = PointNet_Concatenator()
        self.decoder = PointNet_Decoder(input_dim=point_dim+self.encoder.output_shape[0], point_dim=point_dim, use_bias=use_bias)

    def forward(self, x):
        shape_descriptor = self.encoder(x)
        x = self.concatenator(x,shape_descriptor)
        x = self.decoder(x)
        return x
    
if __name__ == '__main__':

    #Test denoiser
    denoiser = PointNet_Denoiser(use_bias=True)
    print('Denoiser parameters:',count_parameters(denoiser))
    print('Denoiser parameters requiring grad:',count_parameters_requiring_grad(denoiser))
    x = torch.rand(32,3,2048)
    y = denoiser(x)
    print('Input shape:',x.shape)
    print('Output shape:',y.shape)
    print('Output shape should be:',x.shape)