import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torchvision import datasets, transforms
from resnet18 import ResNet,ResidualBlock
from vgg16 import VGG16


BATCH_SIZE=10 # 批次大小
EPOCHS=200 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def unpickle(file):
    with open(file, 'rb') as fo: 
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 3,32x32
        self.conv1=nn.Conv2d(3 ,10,3) # 30x30
        self.pool = nn.MaxPool2d(2,2) # 15x15
        self.conv2=nn.Conv2d(10,20,3) # 13x13
        self.fc1 = nn.Linear(20*13*13,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24 
        out = F.relu(out)
        out = self.pool(out)  #12 
        out = self.conv2(out) #10 
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out 
dict = [[],[],[],[],[],[]]

dict[0] = unpickle('Cifar/cifar-10-batches-py/data_batch_1')
dict[1] = unpickle('Cifar/cifar-10-batches-py/data_batch_2')
dict[2] = unpickle('Cifar/cifar-10-batches-py/data_batch_3')
dict[3] = unpickle('Cifar/cifar-10-batches-py/data_batch_4')
dict[4] = unpickle('Cifar/cifar-10-batches-py/data_batch_5')
dict[5] = unpickle('Cifar/cifar-10-batches-py/test_batch')

#model = Net().to(DEVICE)
model = ResNet(ResidualBlock).to(DEVICE)
#model = VGG16().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
model.train()

def process_train(num):
    data = []
    target = []
    for j in range(1000):
        batch_data = []
        batch_target = []
        for i in range(BATCH_SIZE):
            sub_data = dict[num][b'data'][i+j*BATCH_SIZE].reshape(3,32,32)
            sub_target = dict[num][b'labels'][i+j*BATCH_SIZE]
            batch_data.append(sub_data)
            batch_target.append(sub_target)
        data.append(batch_data)
        target.append(batch_target)
    return data,target

data,target = process_train(0)
data1,target1 = process_train(1)
data2,target2 = process_train(2)
data3,target3 = process_train(3)
data4,target4 = process_train(4)

test_data,test_target = process_train(5)

def train(inputdata,inputtarget):
    for i in range(1000):
        batch_data = inputdata[i]
        batch_target = inputtarget[i]
        batch_target = torch.tensor(batch_target).to(DEVICE)
        target_t = torch.tensor(batch_target)  
        data_t = torch.tensor(batch_data)
#        print(target_t)
#        print(data_t.size())
#       data_t = data_t.reshape(1,3,32,32)
        data_t = data_t.to(DEVICE)
        data_t = data_t.float()
        data_t = data_t/255.0
        data_t = (data_t - 0.5)/0.5
        optimizer.zero_grad()
        out = model(data_t)
        loss = criterion(out, target_t)
        loss.backward()
        optimizer.step() 
        if i%1000 ==0:
            print(loss.item())
            


for i in range(20):
    train(data,target)
    train(data1,target1)
    train(data2,target2)
    train(data3,target3)
    train(data4,target4)



model.eval()
corrent_nums = 0
for i in range(1000):
    batch_data = test_data[i]
    batch_target = test_target[i]
#    print(batch_data)
    
    batch_target = torch.tensor(batch_target).to(DEVICE)
    target_t = torch.tensor(batch_target)
    data_t = torch.tensor(batch_data)
    data_t = data_t.to(DEVICE)
    data_t = data_t.float()
    data_t = data_t/255.0
    data_t = (data_t - 0.5)/0.5
    output = model(data_t)
    _, predicted = torch.max(output.data, 1)
    pred = output.max(1, keepdim=True)[1]
    
#    print(predicted)
#    print(target_t)
    corrent_nums  += (predicted == target_t).sum().item()
#    for j in range(4):
#        if predicted[j] == target_t[j]:
#           corrent_nums = corrent_nums + 1

print(corrent_nums)


'''

for j in range(20):

    for i in range(100):
        batch_data = data[i]
        batch_target = target[i]
        batch_target = torch.tensor(batch_target).to(DEVICE)
        target_t = torch.tensor(batch_target)  
        data_t = torch.tensor(batch_data)
#       print(batch_data)
#       print(target_t.size())
#       data_t = data_t.reshape(1,3,32,32)
        data_t = data_t.to(DEVICE)
        data_t = data_t.float()
        optimizer.zero_grad()
        out = model(data_t)
        loss = F.nll_loss(out, target_t)
        loss.backward()
        optimizer.step() 
        print(loss)

'''
 

'''
print(dict[0][b'data'][1])
data = dict[0][b'data'][1].reshape(3,32,32)
data_t = torch.tensor(data)
data_t = data_t.reshape(1,3,32,32)
data_t = data_t.to(DEVICE)
data_t = data_t.float()
print(data_t.size())
out = model(data_t)
'''
