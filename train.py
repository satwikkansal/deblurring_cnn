import tensorflow as tf
import numpy as np

import os
import glob
import random

from PIL import Image

import input_pipeline
import model

train_dir = './data/train/'
test_dir = './data/test/'

# basic parameters for easy setup
TRAINING_DATASET_SIZE = 2000
TESTING_DATASET_SIZE = 200

# training parameters
EPOCHS = 50
BATCH_SIZE = 5
LEARNING_RATE = 0.0001


def shuffle_data(features, labels):
    assert (len(features)==len(labels)),"Input Vector size is not equal to output vector size"

    shuffled_idx = range(len(features))
    random.shuffle(shuffled_idx)
    
    shuffled_features = [features[i] for i in shuffled_idx]
    shuffled_labels = [labels[i] for i in shuffled_idx]

    return shuffled_features,shuffled_labels

# get feature and label filenames from the directory
def get_filenames():
    features =  glob.glob(train_dir + '*_blur.png')
    labels = glob.glob(train_dir + '*_orig.png')
    
    features.sort()
    labels.sort()

    features , labels = shuffle_data(features,labels)

    features = features[:TRAINING_DATASET_SIZE]
    labels = labels[:TRAINING_DATASET_SIZE]
    print features[0],labels[0]
    return features,labels

def setup_tensorflow():
    config = tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config=config)
    
    with sess.graph.as_default():
        tf.set_random_seed(0)

    random.seed(0)

    return sess

def train_neural_network():
    
    sess = setup_tensorflow()


    ffilenames , lfilenames = get_filenames()
    features , labels = input_pipeline.get_files(sess,ffilenames , lfilenames,BATCH_SIZE)
    
    # get outputs and variable lists 
    output , cnn_var_list = model.create_model(sess,features,labels)
   
    # get loss and minimize operations
    cnn_loss = model.create_cnn_loss(output, labels)
    (global_step , learning_rate, cnn_minimize) = model.create_optimizer(cnn_loss, cnn_var_list)
    

    # train the network
    sess.run(tf.global_variables_initializer())

    num_batches = TRAINING_DATASET_SIZE/BATCH_SIZE
    
    for epoch in range(1,EPOCHS+1):
        for batch in range(1,(TRAINING_DATASET_SIZE/BATCH_SIZE) + 1):
            
            # create feed dictionary for passing hyperparameters
            feed_dict = {learning_rate: LEARNING_RATE}

            #create operations list for root nodes of computation graph
            ops = [cnn_minimize , cnn_loss]
            _ , loss = sess.run(ops, feed_dict=feed_dict)

            print ("Epoch : " + str(epoch) + "/" + str (EPOCHS) + " , Batch : " + str(batch) + "/" + str(num_batches) + " completed; Loss " + str(loss))
            
            if batch%SUMMARY_PERIOD == 0:
                # save model progress and save output images for this batch
                

        

train_neural_network() 
        
