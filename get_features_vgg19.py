#!/usr/bin/env python

#feature extraction code source: https://github.com/rnoxy/cifar10-cnn

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras.layers import Input, Dense, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K

tf.disable_v2_behavior()

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

n_testing = X_test.shape[0]

y_test  = y_test.flatten()

print( X_test.shape, y_test.shape )

from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50    import ResNet50
from keras.applications.vgg16        import VGG16
from keras.applications.vgg19        import VGG19

network_names = [ 'vgg19' ]
print("Available networks = ", network_names)
cnnid = int( input("Please choose the CNN network [0-{n}]: ".format(n=len(network_names)-1)) )
selected_network = network_names[cnnid]
print("Selected network: ", selected_network)

input_shape = {
    'vgg19'   : (224,224,3)
}[selected_network]

def create_model_vgg19():
    tf_input = Input(shape=input_shape)
    model = VGG19(input_tensor=tf_input, include_top=False)
    output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, output_pooled )

create_model = {
    'vgg19'    : create_model_vgg19
}[selected_network]

# tensorflow placeholder for batch of images from CIFAR10 dataset
batch_of_images_placeholder = tf.placeholder("uint8", (None, 32, 32, 3))

batch_size = {
    'vgg19'    : 16
}[selected_network]

# Inception default size is 299x299
tf_resize_op = tf.image.resize_images(batch_of_images_placeholder, (input_shape[:2]), method=0)


# data generator for tensorflow session
from keras.applications.inception_v3 import preprocess_input as incv3_preprocess_input
from tensorflow.keras.applications.resnet50     import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16        import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19        import preprocess_input as vgg19_preprocess_input

preprocess_input = {
    'vgg19'   : vgg19_preprocess_input
}[selected_network]

def data_generator(sess,data,labels):
    def generator():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        while True:
            batch_of_images_resized = sess.run(tf_resize_op, {batch_of_images_placeholder: data[start:end]})
            batch_of_images__preprocessed = preprocess_input(batch_of_images_resized)
            batch_of_labels = labels[start:end]
            start += batch_size
            end   += batch_size
            if start >= n:
                start = 0
                end = batch_size
            yield (batch_of_images__preprocessed, batch_of_labels)
    return generator

with tf.Session() as sess:
    # setting tensorflow session to Keras
    K.set_session(sess)
    # setting phase to training
    K.set_learning_phase(0)  # 0 - test,  1 - train

    model = create_model()

    data_test_gen = data_generator(sess, X_test, y_test)
    ftrs_testing = model.predict_generator(data_test_gen(), n_testing/batch_size, verbose=1)

    features_testing  = np.array( [ftrs_testing[i].flatten()  for i in range(n_testing )] )
    print(features_testing.shape)

    np.savez_compressed("features/CIFAR10_{}-keras_features.npz".format(selected_network), \
                    features_testing=features_testing,   \
                    labels_testing=y_test)
