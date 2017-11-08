import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

# Class model for CNN
class Model:

    def __init__(self,name,features):
        self.name = name
        self.outputs = [features]


    def get_output(self):
        return self.outputs[-1]
    

    def get_layer_str(self , layer =None):
        if layer is None:
            layer = self.get_num_layers()

        return '%s_L%03d' % (self.name, layer+1)
    
    def get_num_layers(self):
        return len(self.outputs)

    def get_num_inputs(self):
        return self.get_output().get_shape()[-1]

        
    def add_conv2d(self , kernel_size=1,output_channels=32, stride=1 , stddev_factor= 0.1):
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch , width , height , channels)"

        with tf.variable_scope(self.get_layer_str()):
            input_channels = self.get_num_inputs()
         
            weight = tf.get_variable('weight' ,shape=[kernel_size,kernel_size,input_channels,output_channels],
                    initializer=tf.contrib.layers.xavier_initializer())

            # Compute the decay term
            weight_decay = tf.multiply(tf.nn.l2_loss(weight),FLAGS.beta)
            tf.add_to_collection('losses',weight_decay)

            out = tf.nn.conv2d(self.get_output() , weight , strides=[1,stride,stride,1],padding='SAME')

            initb = tf.constant(0.0 , shape=[output_channels])
            bias = tf.get_variable('bias' , initializer=initb)
            
            tf.summary.histogram("weight",weight)
            tf.summary.histogram("bias", bias)

            out = tf.nn.bias_add(out,bias);

        self.outputs.append(out)
        return self
    

    def add_batch_norm(self , mean=0.1 , variance=0.01 , alpha=0.1, beta=0.1):
        with tf.variable_scope(self.get_layer_str()):
            out = tf.nn.batch_normalization(self.get_output(), mean, variance , alpha , beta, 0.00001) 

        self.outputs.append(out);
        return self
    
    def add_relu(self):
        with tf.variable_scope(self.get_layer_str()):
            out = tf.nn.relu(self.get_output())

        self.outputs.append(out)

def convolutional_nn(sess, features , labels ):

    old_vars = tf.global_variables()

    model = Model('CNN' , features)
    
    # define layer_list as tuples of layers where each tuple has values (kernel_size,output_channels)
    layer_list = [(19,128),(1,320),(1,320),(1,320),(3,128),(1,512),(5,128),(5,128),(3,128),(5,128),(5,128),(1,256),(7,64),(7,3)]
    
    # add conv layers 
    for layer in layer_list[:-1]:
       model.add_conv2d(layer[0],layer[1])
       model.add_relu()
    
    # add the final convolutional layer
    model.add_conv2d(layer_list[-1][0],layer_list[-1][1])
    
    # collect all variables for optimization
    new_vars = tf.global_variables();
    cnn_vars = list(set(new_vars)-set(old_vars))

    return model.get_output() , cnn_vars 


# function to create the deep cnn model
def create_model(sess,features,labels):
    rows = int(features.get_shape()[1])
    cols = int(features.get_shape()[2])
    channels = int(features.get_shape()[3])

    test_input = tf.placeholder(tf.float32, shape=[FLAGS.BATCH_SIZE,rows,cols,channels])
    test_label = tf.placeholder(tf.float32, shape=[FLAGS.BATCH_SIZE,rows,cols,channels])
    
    with tf.variable_scope('cnn') as scope:
        output,cnn_vars = convolutional_nn(sess,features,labels)
        scope.reuse_variables()
        test_output, _  = convolutional_nn(sess,test_input,test_label)


    return output,cnn_vars,test_input,test_label,test_output

def create_cnn_loss(cnn_output , labels):

    # we use euclidean loss function and weight decay as regularizer
    euclidean_loss = tf.losses.mean_squared_error(labels , cnn_output)
    tf.add_to_collection('losses' ,euclidean_loss)
    
    return tf.add_n(tf.get_collection('losses'),name='total_loss')


def create_optimizer(cnn_loss , cnn_vars_list):
    
    global_step = tf.Variable(0 , dtype=tf.int64 ,  trainable=False, name='global_step')
    learning_rate = tf.placeholder(dtype=tf.float32 , name='learning_rate')

    cnn_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='cnn_optimizer')
    cnn_minimize = cnn_optimizer.minimize(cnn_loss, var_list=cnn_vars_list, name='cnn_loss_minimize' , global_step=global_step)

    return (global_step,learning_rate,cnn_minimize) 


