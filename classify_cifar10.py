import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torchvision import datasets, transforms
from vgg16 import VGG16
import torchvision


BATCH_SIZE=100 # 批次大小
EPOCHS=200 # 总共训练批次
LR = 0.01 #学习率
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
def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

#model = Net().to(DEVICE)
#model = ResNet(ResidualBlock).to(DEVICE)
model = VGG16().to(DEVICE)
#optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()
model.train()

def process_train(num):
    data = []
    target = []
    for j in range((int(10000/BATCH_SIZE))):
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

train_transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset=torchvision.datasets.CIFAR10(root='data/',train=True,transform=train_transform,download=False)


test_transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


test_dataset=torchvision.datasets.CIFAR10(root='data/',train=False,transform=test_transform,download=False)


trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50*len(trainloader), gamma=0.1)

print("trainloader len: ", len(train_dataset))
print("testloader len: :", len(test_dataset))



def train(trainloader):
    model.train()
    for i,data in enumerate(trainloader, 0):
        batch_data, batch_target = data
        batch_data, batch_target =batch_data.to(DEVICE), batch_target.to(DEVICE)

        optimizer.zero_grad()
        out = model(batch_data)
        loss = criterion(out, batch_target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i%((int(10000/BATCH_SIZE))) ==0:
            print('loss %.4f ' %loss.item(),"lr:", optimizer.param_groups[0]['lr'])

def test(testloader):
    model.eval()
    corrent_nums = 0
    total = 0
    for i,data in enumerate(testloader, 0):
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        corrent_nums += (predicted == labels).sum().item()
    print(corrent_nums, total)




for i in range(EPOCHS):
    print(str(i)+'/'+str(EPOCHS))
    train(trainloader)
    if i%10==0 and i!=0:
        test(testloader)
        model_name = "./weights/vgg11_dropout_" + str(i) + ".pth"
        model.save_weights(model_name)



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
