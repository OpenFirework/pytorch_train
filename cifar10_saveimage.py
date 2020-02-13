import pickle
import cv2 as cv
import numpy as np
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dict = unpickle('Cifar/cifar-10-batches-py/data_batch_1')
print(len(dict[b'data']))
print(dict[b'data'][1])
print(dict[b'labels'][1])
img = [0 for i in range(32*32*3)]
for i in range(32*32):
   img[2+3*i] = dict[b'data'][1][i]
   img[1+3*i] = dict[b'data'][1][i+32*32]
   img[0+3*i] = dict[b'data'][1][i+32*32*2]
#print(img)
img = np.array(img).reshape(32,32,3)
cv.imwrite('test.png',img)

