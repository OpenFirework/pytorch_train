# pytorch_train

--- 
Lenet的准确率约为53%     

修改后的VGG16准确率约为76%（训练时batchsize设为100，迭代20次），后续可做实验再进行优化（不加BN层的话VGG16的网络不会收敛）  

resnet18的准确率也为约76%（训练时batchsize设为100），不同batchsize对正确率会有影响，设batchsize为4时准确率可到84% 

