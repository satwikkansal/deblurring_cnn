import tensorflow as tf
import numpy as np

import os
import glob
import random

from PIL import Image

import input_pipeline
import model

FLAGS = tf.app.flags.FLAGS 

train_dir = './data/train/'
test_dir = './data/test/'
output_dir = './output/'

# basic parameters for easy setup
TRAINING_DATASET_SIZE = 2000
TESTING_DATASET_SIZE = 200

# training parameters
EPOCHS = 50
BATCH_SIZE = 5
LEARNING_RATE = 0.0001

# Logging Parameters
SUMMARY_PERIOD=10

tf.app.flags.DEFINE_integer('BATCH_SIZE', 5 , 'Batch size for training and testing') 

def shuffle_data(features, labels):
    assert (len(features)==len(labels)),"Input Vector size is not equal to output vector size"

    shuffled_idx = range(len(features))
    random.shuffle(shuffled_idx)
    
    shuffled_features = [features[i] for i in shuffled_idx]
    shuffled_labels = [labels[i] for i in shuffled_idx]

    return shuffled_features,shuffled_labels

# get feature and label filenames from the train directory
def get_filenames():
    features =  glob.glob(train_dir + '*_blur.png')
    labels = glob.glob(train_dir + '*_orig.png')
    
    features.sort()
    labels.sort()

    features , labels = shuffle_data(features,labels)

    features = features[:TRAINING_DATASET_SIZE]
    labels = labels[:TRAINING_DATASET_SIZE]
    return features,labels

def get_test_filenames():
    features =  glob.glob(test_dir + '*_blur.png')
    labels = glob.glob(test_dir + '*_orig.png')
    
    features.sort()
    labels.sort()

    features , labels = shuffle_data(features,labels)

    features = features[:TESTING_DATASET_SIZE]
    labels = labels[:TESTING_DATASET_SIZE]

    return features,labels


def setup_tensorflow():
    config = tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config=config)
    
    with sess.graph.as_default():
        tf.set_random_seed(0)

    random.seed(0)
    return sess

def _save_image_batch(epoch , batch_number ,  image_batch):
    # convert float to int type array
    image_batch = image_batch.astype(int)        
    batch_size = image_batch.shape[0]
    save_dir = os.path.join(output_dir,"Epoch_"+str(epoch),"Batch_"+str(batch_number/10))
    
    # make directory if not exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for image_idx in range(batch_size):
        image = image_batch[image_idx]
        image = Image.fromarray(image,mode='RGB')
        image.save(os.path.join(save_dir,"Image_"+str(image_idx)+".png"))


def train_neural_network():
    
    sess = setup_tensorflow()

    train_feature_filenames , train_label_filenames = get_filenames()
    test_feature_filenames , test_label_filenames = get_test_filenames()
    
    train_features , train_labels = input_pipeline.get_files(sess,train_feature_filenames,train_label_filenames)
    test_features , test_labels = input_pipeline.get_files(sess,test_feature_filenames, test_label_filenames)
    
        
    # get outputs and variable lists 
    output , cnn_var_list, test_input, test_label, test_output = model.create_model(sess,train_features,train_labels)
   
    # get loss and minimize operations
    cnn_loss = model.create_cnn_loss(output, train_labels)
    (global_step , learning_rate, cnn_minimize) = model.create_optimizer(cnn_loss, cnn_var_list)
    

    # train the network
    sess.run(tf.global_variables_initializer())

    # cache test features and labels so we can monitor the progress
    test_feature_batch , test_label_batch = sess.run([test_features,test_labels])

    num_batches = TRAINING_DATASET_SIZE/FLAGS.BATCH_SIZE
    
    for epoch in range(1,EPOCHS+1):
        for batch in range(1,(TRAINING_DATASET_SIZE/FLAGS.BATCH_SIZE) + 1):
            
            # create feed dictionary for passing hyperparameters
            feed_dict = {learning_rate: LEARNING_RATE}

            #create operations list for root nodes of computation graph
            ops = [cnn_minimize , cnn_loss]
            _ , loss = sess.run(ops, feed_dict=feed_dict)

            print ("Epoch : " + str(epoch) + "/" + str (EPOCHS) + " , Batch : " + str(batch) + "/" + str(num_batches) + " completed; Loss " + str(loss))
            
            if batch%SUMMARY_PERIOD == 0:
                # save model progress and save output images for this batch
                feed_dict = {test_input:test_feature_batch, test_label:test_label_batch}
                output_batch = sess.run(test_output , feed_dict=feed_dict)
                
                # save the output images
                _save_image_batch(epoch,batch,output_batch)
                print "Image batch saved!!"


train_neural_network() 
        
