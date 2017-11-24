import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from sklearn.utils import shuffle
import project_tests as tests
import math
import argparse
from moviepy.editor import VideoFileClip
import scipy.misc
import numpy as np


#Global Variables
tensorboard = False #Save temp files for tensorboard
trainning = True #Trainning Enable for the network

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#dowload vgg
#https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #vgg tensor names
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    #load model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    #recover tensors
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_tensor, keep_tensor, layer3_tensor, layer4_tensor, layer7_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    #Regularizer parameters
    alfa = 1e-3
    #Kernel Initializer Parameters
    mu = 0
    sigma = 0.1 

    #layers
    output1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),\
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),\
        #kernel_initializer= tf.contrib.layers.xavier_initializer(uniform=True),
        activation = None,\
        name='encode_1')

    output1 = tf.layers.conv2d_transpose(output1, num_classes, 4, 2, padding='same',\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),\
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),\
        #kernel_initializer= tf.contrib.layers.xavier_initializer(uniform=True),
        activation = None,\
        name='decode_1')
    output1 = tf.nn.elu(output1)

    pool4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),\
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),\
        #kernel_initializer= tf.contrib.layers.xavier_initializer(uniform=True),
        activation = None,\
        name='encode_pool_4')

    output2 = tf.add(output1, pool4, name='add_1')
    output2 = tf.layers.conv2d_transpose(output2, num_classes, 4, 2, padding='same',\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),\
        kernel_initializer=tf.random_normal_initializer(mean=mu, stddev=sigma),\
        #kernel_initializer= tf.contrib.layers.xavier_initializer(uniform=True),
        activation = None,\
        name='decode_pool_4')
    output2 = tf.nn.elu(output2)

    pool3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),\
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),\
        #kernel_initializer= tf.contrib.layers.xavier_initializer(uniform=True),
        activation = None,\
        name='encode_pool_3')

    output3 = tf.add(output2, pool3, name='add_2')
    output3 = tf.layers.conv2d_transpose(output3, num_classes, 16, 8, padding='same',\
        kernel_regularizer=tf.contrib.layers.l2_regularizer(alfa),\
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),\
        #kernel_initializer= tf.contrib.layers.xavier_initializer(uniform=True),
        activation = None,\
        name='decode_pool_3')
    output3 = tf.nn.elu(output3)

    return output3
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    #Reshape both the input image and the ground truth image
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')    
    correct_label = tf.reshape(correct_label, (-1, num_classes), name='correct_label')
    #define loss function and optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label), name='mean')
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name='Adam')
    training_operation = optimizer.minimize(cross_entropy_loss, name='Minimize')    
    
    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)

def summary(cross_entropy_loss):
    """
    Keep track of data for the TensorBoard
    :param loss: TF Tensor of cross entropy loss
    :return: TF tensor summary operator 
    """

    #for tensorboard
    tf.summary.scalar("cost", cross_entropy_loss)
    summary_op = tf.summary.merge_all()

    return summary_op

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    #initialize variables    
    sess.run(tf.global_variables_initializer())

    #save model at the end
    if(epochs > 1): #to avoid bugs with test function
        saver = tf.train.Saver()

    #user choose to use tensorboard
    if tensorboard:
        writer = tf.summary.FileWriter('/tmp/tensorflow', graph=tf.get_default_graph())
        summary_op = summary(cross_entropy_loss)

    l_rate = 0.001
    acc_loss =  1000000
    for i in range(epochs):
        print("Run EPOCH " + str(i))
        j = 0
        m_loss = 1000000
        a_loss =[]
        for batch_x, batch_y in get_batches_fn(batch_size):
            print("Processing " + str(j*batch_size) + " images")
            j += 1
            #Run optmizer
            if not tensorboard:
                _, loss = sess.run([train_op, cross_entropy_loss],\
                    feed_dict={input_image: batch_x,\
                        correct_label: batch_y,\
                        learning_rate: l_rate,\
                        keep_prob: 0.5})
            else:
                _, loss, smm = sess.run([train_op, cross_entropy_loss, summary_op],\
                    feed_dict={input_image: batch_x,\
                        correct_label: batch_y,\
                        learning_rate: l_rate,\
                        keep_prob: 0.5})
            a_loss.append(loss)
            print("LOSS " + str(loss))
            #save summary data for tensorboard every 32 images
            if tensorboard and (j*batch_size) % 32 == 0:
                print("SAVE TENSORBOARD")
                writer.add_summary(smm, (300*i) + j) 
        m_loss = np.mean(np.array(a_loss))
        if acc_loss == 1000000:
            acc_loss = m_loss
        l = m_loss - acc_loss
        print("Local / Global Loss " + str(m_loss) + " / " + str(acc_loss))
        if l > 0:
            #l_rate = l_rate /10
            print("Change Learning Rate: " + str(l_rate))
        acc_loss = m_loss
        #save model (checkpoint)        
        if(epochs > 1): #to avoid bugs with test function
            print("SAVING MODEL")
            saver.save(sess, "data/model/model.ckpt")

tests.test_train_nn(train_nn)
 

def create_video(video, sess, logits, keep_prob, image_input, image_shape):
    def pipeline(image):
        image = scipy.misc.imresize(image, image_shape)
        im_softmax = sess.run([tf.nn.softmax(logits)],\
            {keep_prob: 1.0, image_input: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        return np.array(street_im)

    #process video
    new_video = video.fl_image(pipeline)
    #save new video
    new_video.write_videofile('result.mp4')


def run():
    #global variables
    global tensorboard, trainning
    #Arguments
    parser = argparse.ArgumentParser(description='Semantic Segmentation Trainning')
    parser.add_argument('-tb', '--tensorboard', action='store_true', help='Save data for TensorBoard') 
    parser.add_argument('-to', '--trainning_off',  action='store_false', help='Set Trainning OFF for the network')
    parser.add_argument('-v', '--video',  default=None, type=str, help='Apply Segmentation FCN network over the video')
    args = parser.parse_args()
    tensorboard = args.tensorboard
    trainning = args.trainning_off
    video = args.video

    #Tranning values
    epochs = 8
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    batch_size = 4
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)    

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(
            os.path.join(data_dir, 'data_road/training'), image_shape)
            
        # Build NN using load_vgg, layers, and optimize function
        # Load VGG
        input_tensor, keep_prob_tensor, layer3_tensor, layer4_tensor, layer7_tensor = load_vgg(sess, vgg_path)
        # Transfor into a FCN
        output = layers(layer3_tensor, layer4_tensor, layer7_tensor, num_classes)
        # Load Optmizer
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        
        # Train NN using the train_nn function
        
        if trainning:
            print("Trainning Model")
            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_tensor,
                 correct_label, keep_prob_tensor, learning_rate)        
        
            # Save inference data using helper.save_inference_samples
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_tensor)
        else: 
            print("Loading pre-trained model")
            #load model
            saver = tf.train.Saver()
            saver.restore(sess, "./data/model/model.ckpt")

        #Apply the trained model to a video
        if video is not None:
            print("Loading video " + video)
            print("Creating Video, saved at result.mp4")
            #clip = VideoFileClip('driving_10.mp4')
            clip = VideoFileClip(video)
            create_video(clip, sess, logits, keep_prob_tensor, input_tensor, image_shape)

if __name__ == '__main__':
    run()
