import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        
        # 3 * 32 * 32
        self.conv1_1 = nn.Conv2d(3, 64, 3) # 64 * 30 * 30
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 30* 30
        self.bn1_2 = nn.BatchNorm2d(64)
#        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112
        
        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 28 * 28
        self.bn2_1 = nn.BatchNorm2d(128)
#        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 28 * 28
#        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 15 * 15
        
        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 13 * 13
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 13 * 13
        self.bn3_2 = nn.BatchNorm2d(256)
 #       self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 26 * 26
#        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28
        
        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 11 * 11
        self.bn4_1 = nn.BatchNorm2d(512)
  #      self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 24 * 24
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 11 * 11
        self.bn4_3 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 6 * 13
        
        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 11 * 11
        self.bn5_1 = nn.BatchNorm2d(512)
   #     self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 11 * 11
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 11 * 11
        self.bn5_3 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 6 * 6
        
        # view
        
        self.fc1 = nn.Linear(512 * 6 * 6, 360)
#        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(360, 10)
        # softmax 1 * 1 * 1000
        
    def forward(self, x):
        
        # x.size(0)即为batch_size
        in_size = x.size(0)
        
        out = self.conv1_1(x) # 222
        out = self.bn1_1(out)
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = self.bn1_2(out)
        out = F.relu(out)
#        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = self.bn2_1(out)
        out = F.relu(out)
        #out = self.conv2_2(out) # 110
        #out = F.relu(out)
 #       out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = self.bn3_1(out)
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = self.bn3_2(out)
        out = F.relu(out)
        #out = self.conv3_3(out) # 54
        #out = F.relu(out)
#        out = self.maxpool3(out) # 28
        
        out = self.conv4_1(out) # 26
        out = self.bn4_1(out)
        out = F.relu(out)
       # out = self.conv4_2(out) # 26
       # out = F.relu(out)
        out = self.conv4_3(out) # 26
        out = self.bn4_3(out)
        out = F.relu(out)
        out = self.maxpool4(out) # 14
        
        out = self.conv5_1(out) # 12
        out = F.relu(out)
        #out = self.conv5_2(out) # 12
        #out = F.relu(out)
        out = self.conv5_3(out) # 12
        out = F.relu(out)
        out = self.maxpool5(out) # 7
        
        # 展平
        out = out.view(in_size, -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        #out = self.fc2(out)
        #out = F.relu(out)
        out = self.fc3(out)
#        out = F.log_softmax(out, dim=1)
        return out
