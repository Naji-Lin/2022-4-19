import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from multilayer_perceptron import MultilayerPerceptron

data = pd.read_csv('./data/MNIST_CSV格式/mnist_train.csv')
# 展示数据
numbers = 25
num_cell = math.ceil(math.sqrt(numbers))
plt.figure(figsize=(10,10))
for plot in range(numbers):
    picture = data[plot:plot+1].values
    label = picture[0][0]
    pixel = picture[0][1:]
    picture_size = int(math.sqrt(pixel.shape[0]))
    frame = pixel.reshape((picture_size,picture_size))
    plt.subplot(num_cell,num_cell,plot+1)
    plt.imshow(frame,cmap='Greys')
    plt.title(label)
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
train_data = train_data.values
test_data = test_data.values

num_training_examples = 700

pixel_train = train_data[:num_training_examples,1:]
label_train = train_data[:num_training_examples,[0]]
pixel_test = test_data[:,1:]
label_test = test_data[:,[0]]
layers=[784,25,10]
normalize_data = True
max = 300
alpha = 0.1

multilayerPerceptron = MultilayerPerceptron(pixel_train,label_train,layers,normalize_data)
(ws,loss_history) = multilayerPerceptron.train(max,alpha)
plt.plot(range(len(loss_history)),loss_history)
plt.xlabel('Gradient descent steps')
plt.ylabel('loss')
plt.show()

result_train = multilayerPerceptron.predict(pixel_train)
result_test = multilayerPerceptron.predict(pixel_test)

train_percent = np.sum(result_train == label_train) / label_train.shape[0] * 100
test_percent = np.sum(result_test == label_test) / label_test.shape[0] * 100
print('训练集准确率',train_percent)
print('测试集准确率',test_percent)

numbers_to_display = 64
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(15, 15))
for plot_index in range(numbers_to_display):
    digit_label = label_test[plot_index, 0]
    digit_pixels = pixel_test[plot_index, :]
    predicted_label = result_test[plot_index][0]
    image_size = int(math.sqrt(digit_pixels. shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))
    color_map = 'Greens' if predicted_label == digit_label else 'Reds'
    plt.subplot(num_cells, num_cells, plot_index + 1)
    plt.imshow(frame, cmap=color_map)
    plt.title(predicted_label)
    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False)
plt.subplots_adjust (hspace=0.5,wspace=0.5)
plt.show( )



