import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
from tqdm import tqdm

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13*13*8, 10) # Input Lens and number of Nodes/Classes


def forward( image, label ):
    '''
    Completes a forward pass of the cnn and calculated the accuracy and loss

    params:
    --------
    image : is a 2d numpy array
    label : is a digit
    '''
    # Trivial Image preprocessing
    out = conv.forward(( image/255 ) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculating Loss and Accuracy
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

print(' ------------ MNIST CNN Initialized  --------------- ')

# Uncomment when using second tqdm loop
# progress_bar = tqdm(enumerate(zip(test_images, test_labels)), total=len(test_images))

loss = 0
num_correct = 0

for i, (im, label) in enumerate(zip(test_images, test_labels)):
 
    # Forward Pass
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # print stats every 100 steps.
    if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, num_correct))

        loss = 0
        num_correct = 0

# Works correctly btw
# for i, (im, label) in progress_bar:
#     # forward pass
#     _,l, acc = forward(im, label)
#     loss += 1
#     num_correct += acc

#     progress_bar.set_description(f'Loss: {loss / (i + 1):.3f}, Accuracy: {num_correct / (i + 1) * 100:.2f}%')

# print(f'Average Loss: {loss / len(test_images):.3f} | Accuracy: {num_correct / len(test_images) * 100:.2f}%')
