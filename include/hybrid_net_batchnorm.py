import torch
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_16_features = torchvision.models.vgg16_bn(pretrained=True).features
        # First three convolution and pooling cycles have padding of 1 and a kernel size of 3
        self.conv1_1 = vgg_16_features[0]
        self.conv1_1.padding = 1
        self.bn1_1 = vgg_16_features[1]
        self.conv1_2 = vgg_16_features[3]
        self.conv1_2.padding = 1
        self.bn1_2 = vgg_16_features[4]
        self.pool1 = vgg_16_features[6]
        self.pool1.kernel_size=3
        #self.pool1.padding=1
        self.conv2_1 = vgg_16_features[7]
        self.conv2_1.padding = 1
        self.bn2_1 = vgg_16_features[8]
        self.conv2_2 = vgg_16_features[10]
        self.conv2_2.padding = 1
        self.bn2_2 = vgg_16_features[11]
        self.pool2 = vgg_16_features[13]
        self.pool2.kernel_size=3
        #self.pool2.padding=1
        self.conv3_1 = vgg_16_features[14]
        self.conv3_1.padding = 1
        self.bn3_1 = vgg_16_features[15]
        self.conv3_2 = vgg_16_features[17]
        self.conv3_2.padding = 1
        self.bn3_2 = vgg_16_features[19]
        self.conv3_3 = vgg_16_features[21]
        self.conv3_3.padding = 1
        self.bn3_3= vgg_16_features[22]
        self.pool3 = vgg_16_features[23]
        self.pool3.kernel_size=3
        #self.pool3.padding=1

        # For layer 4, the pooling has a stride of 1 so no downsampling occurs
        self.conv4_1 = vgg_16_features[24]
        self.conv4_1.padding = 1
        self.bn4_1= vgg_16_features[25]
        self.conv4_2 = vgg_16_features[27]
        self.conv4_2.padding = 1
        self.bn4_2= vgg_16_features[28]
        self.conv4_3 = vgg_16_features[30]
        self.conv4_3.padding = 1
        self.bn4_3 = vgg_16_features[31]
        self.pool4 = vgg_16_features[33]
        self.pool4.kernel_size=3
        self.pool4.padding=1
        self.pool4.stride=1

        # For layer 5 we begin with atrous convolution
        self.conv5_1 = vgg_16_features[34]
        self.conv5_1.dilation = 2
        self.conv5_1.padding = 2
        self.bn5_1 = vgg_16_features[35]
        self.conv5_2 = vgg_16_features[37]
        self.conv5_2.dilation = 2
        self.conv5_2.padding = 2
        self.bn5_2 = vgg_16_features[38]
        self.conv5_3 = vgg_16_features[40]
        self.conv5_3.dilation = 2
        self.conv5_3.padding = 2
        self.bn5_3 = vgg_16_features[41]
        self.pool5 = vgg_16_features[43]
        self.pool5.stride = 1
        self.pool5.kernel_size=3
        self.pool5.padding=1

        self.isOnCuda = True    
        self.dropout = nn.Dropout2d(inplace=True)       
        self.relu = nn.ReLU(inplace=False)      

    def padForPooling(self, x):
        shape_a = (x.shape[0], x.shape[1], 1, x.shape[3])
        shape_b = (x.shape[0], x.shape[1], x.shape[2]+1, 1)
        if self.isOnCuda:
            return torch.cat((torch.cat((x,torch.zeros(shape_a).cuda()),dim=2),torch.zeros(shape_b).cuda()),dim=3)
        else:
            return torch.cat((torch.cat((x,torch.zeros(shape_a)),dim=2),torch.zeros(shape_b)),dim=3)

    def forward(self, x):
        # First bit is VGG16 unchanged
        
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.padForPooling(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.padForPooling(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.padForPooling(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.relu(x)

#        x = self.padForPooling(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.relu(x)
#        x = self.padForPooling(x)
        x = self.pool5(x)

 
        return x

class Atrous(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        #The fully connected layers are replaced with 4-lane atrous convolution
        self.conv6_l1 = nn.Conv2d(512, 1024, (3,3), dilation = 2, padding = 2)
        self.bn6_l1 = nn.BatchNorm2d(1024)
        self.conv6_l2 = nn.Conv2d(512, 1024, (3,3), dilation = 4, padding = 4)
        self.bn6_l2 = nn.BatchNorm2d(1024)
        self.conv6_l3 = nn.Conv2d(512, 1024, (3,3), dilation = 8, padding = 8)
        self.bn6_l3 = nn.BatchNorm2d(1024)
        self.conv6_l4 = nn.Conv2d(512, 1024, (3,3), dilation = 12, padding = 12)
        self.bn6_l4 = nn.BatchNorm2d(1024)
        self.conv7_l1 = nn.Conv2d(1024, 1024, (1,1))
        self.bn7_l1 = nn.BatchNorm2d(1024)
        self.conv7_l2 = nn.Conv2d(1024, 1024, (1,1))
        self.bn7_l2 = nn.BatchNorm2d(1024)
        self.conv7_l3 = nn.Conv2d(1024, 1024, (1,1))
        self.bn7_l3 = nn.BatchNorm2d(1024)
        self.conv7_l4 = nn.Conv2d(1024, 1024, (1,1))
        self.bn7_l4 = nn.BatchNorm2d(1024)
        self.conv8_l1 = nn.Conv2d(1024, num_classes, (1,1))
        self.conv8_l2 = nn.Conv2d(1024, num_classes, (1,1))
        self.conv8_l3 = nn.Conv2d(1024, num_classes, (1,1))
        self.conv8_l4 = nn.Conv2d(1024, num_classes, (1,1))
        
        self.isOnCuda = True    
        self.dropout = nn.Dropout2d(inplace=False)       
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        
       # Classifier (4-lane atrous convolution)
        x1 = self.conv6_l1(x)
        x1 = self.bn6_l1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.conv6_l2(x)
        x2 = self.bn6_l2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.conv6_l3(x)
        x3 = self.bn6_l3(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x4 = self.conv6_l4(x)
        x4 = self.bn6_l4(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        
        x1 = self.conv7_l1(x1)
        x1 = self.bn7_l1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.conv7_l2(x2)
        x2 = self.bn7_l2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.conv7_l3(x3)
        x3 = self.bn7_l3(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        
        x4 = self.conv7_l4(x4)
        x4 = self.bn7_l4(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        
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


        
