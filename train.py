import tensorflow as tf
import numpy as np
import os
import glob
import png
from PIL import Image
import input_pipeline
import model

train_dir = './data/train/'
test_dir = './data/test/'

BATCH_SIZE = 5
LEARNING_RATE = 0.0001


# get feature and label filenames from the directory
def get_filenames():
    features =  glob.glob(train_dir + '*_blur.png')
    labels = glob.glob(train_dir + '*_orig.png')

    features.sort()
    labels.sort()

    return features,labels

def setup_tensorflow():
    config = tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config)
    
    with sess.graph.as_default():
        tf.set_random_seed(0)

    random.seed(0)

    return sess

def train_neural_network(feature_batch , label_batch):
    
    sess = setup_tensorflow()


    ffilenames , lfilenames = get_filenames()
    features , labels = get_files(sess,ffilenames , lfilenames)
    
    # get outputs and variable lists 
    output , cnn_var_list = create_model(sess,features,labels)
   
    # get loss and minimize operations
    cnn_loss = model.create_cnn_loss(output, labels)
    (global_step , learning_rate, cnn_minimize) = model.create_optimizer(cnn_loss, cnn_var_list)
    

    # train the network
    sess.run(tf.global_variables_initializer())

    done = False

    while not done:
        batch += 1
    
        # create feed dictionary for passing hyperparameters
        feed_dict = {learning_rate: LEARNING_RATE}

        #create operations list for root nodes of computation graph
        ops = [cnn_minimize , cnn_loss]
        _ , loss = sess.run(ops, feed_dict=feed_dict)

         



     
        
