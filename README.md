# Semantic Segmentation Project

This project consists of a TensorFlow implementation of a FCN-8 (full convolutional) network for semantic segmentation. 

The main goal of this project is to develop a Neural Network capable of. 
Figure 1 depicts an example of the input and output images processed by the network.
![alt text][image1]

[//]: # (Image References)

[image1]: images/example.png "Input - Outpu image example."
[image2]: images/resuts.png "Final Segmented image." 

## 1.Access 

The source code for this project is available at [project code](https://github.com/rtsaad/CarND-Semantic-Segmentation).

## 2.Files

The following files are part of this project:
* main.py:  main file that integrates the controller with the simulator;
* test.py: vehicle class which defines the Finite State Machine;
* helper.py: cost_function class to compute the optimun path;


### 2.1 Dependency

This project requires the following packages to work:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

### 2.2 Dataset

This projects use the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). to train and evaluate the neural network. The dataset will be automatically downloaded and extracted into the `data/data_folder` folder the first time the program is executed.

## 3.How to use this project

To run this project, execute the main.py file. 

``` py
python main.py

```
This project offers a few optional flags to help reuse the trained model.
 - -to: disable the trainning pass;
 - -v: process an example video track to test the trained model;
 - -tb: enable tensorboard to visualize the training process (Loss evolution over time and the TensorFlow Graph).


### 3.1 Compiling and Running

The main program can be built and run by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./path_planning
6. Run the Udacity Simulator (./term3_simulator)

## 4 Semantic Segmentation

Semantic segmantion is the task of assigning meaning to part of an object. It helps to separate the image at pixel level and assign each pixel to a target class, such as road. It is also know as scene understanding and it is very relevant to autonomous driving.

### 4.1 Fully Convolution Network

This projecs implements the [FCN-8](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) Fully Convolutional Network developed at Berkeley that take input of arbitrary size and produce correspondingly-sized output with N target classes. The FCN-8 efficiently allows us to train an end-to-end, pixels-to-pixels network for semantic segmenation.

### 4.2 Encoder

The FCN-8 consists of an encoder/decoder network. The first part is a encoder network that adapts the classification [VGGnet](https://arxiv.org/pdf/1409.1556/) network. The initial weights for the encoding par is a pretrained model on the ImageNet for classification. 

The dense (fully connected) layer from the original VGG is replaced by 1-by-1 convolution to preserve the spatial information. 

```python
output1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),
        activation = None,\
        name='encode_1')
```

### 4.3 Decoder

The decoder part of the network upsample the input into the original image size. The output will be a 4-dimensional tensor: batch size, original image size (height/width) and number of classes.

```python
output1 = tf.layers.conv2d_transpose(output1, num_classes, 4, 2, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),
        activation = None,
        name='decode_1')
output1 = tf.nn.elu(output1)
```

The FCN-8 use skip connections to improve the accurace os the segmentations. The skip connections consists of combining previous layers from the VGG with shallow layers.

```python
pool4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
	kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
	kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma), 
        activation = None,
        name='encode_pool_4')
output2 = tf.add(output1, pool4, name='add_1')
```

The complete Decoder Layer is presented below:

```python

   output1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma), 
        activation = None,
        name='encode_1') 

    output1 = tf.layers.conv2d_transpose(output1, num_classes, 4, 2, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma), 
        activation = None,\
        name='decode_1')
    output1 = tf.nn.elu(output1)

    pool4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),
        activation = None,
        name='encode_pool_4')

    output2 = tf.add(output1, pool4, name='add_1')
    output2 = tf.layers.conv2d_transpose(output2, num_classes, 4, 2, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),
        activation = None,
        name='decode_pool_4')
    output2 = tf.nn.elu(output2)

    pool3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation = None,
        name='encode_pool_3')

    output3 = tf.add(output2, pool3, name='add_2')
    output3 = tf.layers.conv2d_transpose(output3, num_classes, 16, 8, padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        activation = None,
        name='decode_pool_3')
    output3 = tf.nn.elu(output3)
```

### 4.4 Trainning

The final step is to train the network by assinging each pixel to the appropriate class. First, we reshape both the input (logits) and the ground truth (correct label) images. After, we use the standard cross entropy loss function, with the Ada optimizer, to train the network.


```python
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    correct_label = tf.reshape(correct_label, (-1, num_classes), name='correct_label')
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label), name='mean')
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name='Adam')
    training_operation = optimizer.minimize(cross_entropy_loss, name='Minimize') 
```

### 4.5 Results

Figure 2 depicst the segmented image trainning the FCN-8 network after 8 epochs with batch of size 4. 

![alt text][image2]

