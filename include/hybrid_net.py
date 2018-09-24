import torch
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_16_features = torchvision.models.vgg16(pretrained=True).features
        # First three convolution and pooling cycles have padding of 1 and a kernel size of 3
        self.conv1_1 = vgg_16_features[0]
        self.conv1_1.padding = 1
        self.conv1_2 = vgg_16_features[2]
        self.conv1_2.padding = 1
        self.pool1 = vgg_16_features[4]
        self.pool1.kernel_size=3
        #self.pool1.padding=1
        self.conv2_1 = vgg_16_features[5]
        self.conv2_1.padding = 1
        self.conv2_2 = vgg_16_features[7]
        self.conv2_2.padding = 1
        self.pool2 = vgg_16_features[9]
        self.pool2.kernel_size=3
        #self.pool2.padding=1
        self.conv3_1 = vgg_16_features[10]
        self.conv3_1.padding = 1
        self.conv3_2 = vgg_16_features[12]
        self.conv3_2.padding = 1
        self.conv3_3 = vgg_16_features[14]
        self.conv3_3.padding = 1
        self.pool3 = vgg_16_features[16]
        self.pool3.kernel_size=3
        #self.pool3.padding=1
        self.isOnCuda = True
        self.dropout1 = nn.Dropout2d(inplace=False) 
        self.dropout2 = nn.Dropout2d(inplace=False)       

        self.relu = nn.ReLU(inplace=False)

    def padForPooling(self, x):
        shape_a = (x.shape[0], x.shape[1], 1, x.shape[3])
        shape_b = (x.shape[0], x.shape[1], x.shape[2]+1, 1)
        if self.isOnCuda:
            return torch.cat((torch.cat((x,torch.zeros(shape_a).cuda()),dim=2),torch.zeros(shape_b).cuda()),dim=3)
        else:
            return torch.cat((torch.cat((x,torch.zeros(shape_a)),dim=2),torch.zeros(shape_b)),dim=3)

    def forward(self, x):
#        dropout = nn.Dropout2d(inplace=True)       
#        relu = nn.ReLU(inplace=True)      
        # First bit is VGG16 unchanged
        
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.padForPooling(x)
        x = self.pool1(x)
       # x = self.dropout1(x)
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.padForPooling(x)
        x = self.pool2(x)
        #x = self.dropout2(x)
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.padForPooling(x)
        x = self.pool3(x)
        return x

class Atrous(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        vgg_16_features = torchvision.models.vgg16(pretrained=True).features
        # For layer 4, the pooling has a stride of 1 so no downsampling occurs
        self.conv4_1 = vgg_16_features[17]
        self.conv4_1.padding = 1
        self.conv4_2 = vgg_16_features[19]
        self.conv4_2.padding = 1
        self.conv4_3 = vgg_16_features[21]
        self.conv4_3.padding = 1
        self.pool4 = vgg_16_features[23]
        self.pool4.kernel_size=3
        self.pool4.padding=1
        self.pool4.stride=1
        
        # For layer 5 we begin with atrous convolution
        self.conv5_1 = vgg_16_features[24]
        self.conv5_1.dilation = 2
        self.conv5_1.padding = 2
        self.conv5_2 = vgg_16_features[26]
        self.conv5_2.dilation = 2
        self.conv5_2.padding = 2
        self.conv5_3 = vgg_16_features[28]
        self.conv5_3.dilation = 2
        self.conv5_3.padding = 2
        self.pool5 = vgg_16_features[30]
        self.pool5.stride = 1
        self.pool5.kernel_size=3
        self.pool5.padding=1
        
        #The fully connected layers are replaced with 4-lane atrous convolution
        self.conv6_l1 = nn.Conv2d(512, 1024, (3,3), dilation = 2, padding = 2)
        self.conv6_l2 = nn.Conv2d(512, 1024, (3,3), dilation = 4, padding = 4)
        self.conv6_l3 = nn.Conv2d(512, 1024, (3,3), dilation = 8, padding = 8)
        self.conv6_l4 = nn.Conv2d(512, 1024, (3,3), dilation = 12, padding = 12)
        self.conv7_l1 = nn.Conv2d(1024, 1024, (1,1))
        self.conv7_l2 = nn.Conv2d(1024, 1024, (1,1))
        self.conv7_l3 = nn.Conv2d(1024, 1024, (1,1))
        self.conv7_l4 = nn.Conv2d(1024, 1024, (1,1))
        self.conv8_l1 = nn.Conv2d(1024, num_classes, (1,1))
        self.conv8_l2 = nn.Conv2d(1024, num_classes, (1,1))
        self.conv8_l3 = nn.Conv2d(1024, num_classes, (1,1))
        self.conv8_l4 = nn.Conv2d(1024, num_classes, (1,1))
        
        self.isOnCuda = True
        self.dropout1_1 = nn.Dropout2d(inplace=False)
        self.dropout1_2 = nn.Dropout2d(inplace=False)       
        self.dropout1_3 = nn.Dropout2d(inplace=False)       
        self.dropout1_4 = nn.Dropout2d(inplace=False)       
        self.dropout2_1 = nn.Dropout2d(inplace=False)       
        self.dropout2_2 = nn.Dropout2d(inplace=False)       
        self.dropout2_3 = nn.Dropout2d(inplace=False)       
        self.dropout2_4 = nn.Dropout2d(inplace=False)       

        self.relu = nn.ReLU(inplace=False)

    # Adds an extra row and column of zeros, so we can do max-pooling with stride 1 
    # and get the same output resolution as input resolution
    def padForPooling(self, x):
        shape_a = (x.shape[0], x.shape[1], 1, x.shape[3])
        shape_b = (x.shape[0], x.shape[1], x.shape[2]+1, 1)
        if self.isOnCuda:
            return torch.cat((torch.cat((x,torch.zeros(shape_a).cuda()),dim=2),torch.zeros(shape_b).cuda()),dim=3)
        else:
            return torch.cat((torch.cat((x,torch.zeros(shape_a)),dim=2),torch.zeros(shape_b)),dim=3)
    
    def forward(self, x):
        #relu = nn.ReLU(inplace=True)
        #dropout = nn.Dropout2d(inplace=False)       
        # Atrous convolution part
        
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)

#        x = self.padForPooling(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.relu(x)
#        x = self.padForPooling(x)
        x = self.pool5(x)

        # Classifier (4-lane atrous convolution)
        x1 = self.conv6_l1(x)
        x1 = self.relu(x1)
        x1 = self.dropout1_1(x1)
        x2 = self.conv6_l2(x)
        x2 = self.relu(x2)
        x2 = self.dropout1_2(x2)
        x3 = self.conv6_l3(x)
        x3 = self.relu(x3)
        x3 = self.dropout1_3(x3)
        x4 = self.conv6_l4(x)
        x4 = self.relu(x4)
        x4 = self.dropout1_4(x4)
        
        x1 = self.conv7_l1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout2_1(x1)
        x2 = self.conv7_l2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout2_2(x2)
        x3 = self.conv7_l3(x3)
        x3 = self.relu(x3)
        x3 = self.dropout2_3(x3)
        x4 = self.conv7_l4(x4)
        x4 = self.relu(x4)
        x4 = self.dropout2_4(x4)
        
        x1 = self.conv8_l1(x1)
        x2 = self.conv8_l2(x2)
        x3 = self.conv8_l3(x3)
        x4 = self.conv8_l4(x4)
        
        # sum fusion
        out = x1 + x2 + x3 + x4

        return out
                        
            
class SegmentationModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vgg=VGG16()
        self.atr=Atrous(num_classes)

  
    def forward(self, x):
        o1=self.vgg(x)
        o2=self.atr(o1)
        return o2


        
