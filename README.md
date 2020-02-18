# pytorch_train

--- 
Lenet的准确率约为53%     

修改后的VGG16准确率约为76%（训练时batchsize设为100，迭代20次），后续可做实验再进行优化（不加BN层的话VGG16的网络不会收敛）不同batchsize对正确率会有影响，设batchsize为4时准确率可到80%  

resnet18的准确率也为约76%（训练时batchsize设为100，同样batchsize训练90次，并加入学习率衰减机制，准确率也可以达到85%），同样不同batchsize对正确率会有影响，设batchsize为4时(训练20次）准确率可到84%,更换为Adam优化算法后（并训练90次，加入学习率衰减机制），可以将准确率提升到85.3%

