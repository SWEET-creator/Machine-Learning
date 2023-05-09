from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
'''
digit=load_digits()
digit.data.shape
plt.matshow(digit.images[0])
digit.images[0]
plt.show()
'''

(train_images, train_labels), (test_images, test_labels)  = load_digits()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Convolution:
    def feedforward(self, input):
    def backprop(self, input):

epoch = 10
for e in range(len(epoch)):
    for ims in train_images:
        Convolution(ims)