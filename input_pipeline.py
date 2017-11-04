import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# get files and features and input streams for 
# tensorflow product graph
def get_files(sess, ffilenames , lfilenames):
    train_feature_input_queue = tf.train.string_input_producer(ffilenames,shuffle=False)
    train_label_input_queue = tf.train.string_input_producer(lfilenames, shuffle=False)

    feature_reader = tf.WholeFileReader()
    label_reader = tf.WholeFileReader()

    
    # read the image from tf.wholeFileReader
    _,feature_images = feature_reader.read(train_feature_input_queue)
    _,label_images = label_reader.read(train_label_input_queue)
    
    # decode to convert into tensors to work in tensorflow
    train_image = tf.image.decode_png(feature_images , channels=3 )
    train_label = tf.image.decode_png(label_images , channels=3 )

    # Completely define the image shape for tensorflow
    timage = tf.reshape(train_image , [300,300,3])
    tlabel = tf.reshape(train_label , [300,300,3])

    # Get the batches from the defined images
    train_image_batch , train_label_batch = tf.train.batch(
            [timage,tlabel],
            batch_size=FLAGS.BATCH_SIZE) 

    # start queue runners and enable coordinator
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess , coord=coord)

    return tf.cast(train_image_batch, tf.float32) , tf.cast(train_label_batch,tf.float32)
