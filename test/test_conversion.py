# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
import caffe

from hypothesis import given, example, settings, unlimited
from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays as gen_arrays


img_strategy = gen_arrays(
    np.uint8,
    (256, 256, 3),
    elements=integers(min_value=0, max_value=255)
)


class ConversionTest(object):
    def __init__(self, def_path, caffemodel_path, data_path):
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256
        self.NUM_CHANNELS = 3

        print('Loading caffe model')
        self.caffe_net = self.load_net(def_path, caffemodel_path)
        
        self.tf_input_tensor, self.tf_model = self.import_model()
        self.tf_sess = tf.Session()
        print('Loading weights into tf model')
        self.tf_model.load(data_path, self.tf_sess)

    def load_net(self, def_path, data_path):
        batch_size = 1
        net = caffe.Net(def_path, data_path, caffe.TEST)
        input_width = net.blobs['data'].width
        input_height = net.blobs['data'].height
        num_channels = net.blobs['data'].channels
        net.blobs['data'].reshape(batch_size, num_channels, input_width, input_height)
        return net

    def import_model(self):
        import kaffe.model as model
        kaffe_model = model.model
        input_node = tf.placeholder(tf.float32,
                                    shape=(None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_CHANNELS))
        return (input_node, kaffe_model({'data': input_node}))

    def close(self):
        self.tf_sess.close()
    
    def execute_example(self, f):
        return f()

    @given(arr = img_strategy)
    @settings(deadline=None, max_examples=50, timeout=unlimited)
    def test_conversion(self, arr):
        imgs = np.array([arr])
        # print(imgs)

        print('Running constant image through tf network')
        tf_output = self.tf_sess.run(self.tf_model.get_output(), feed_dict={self.tf_input_tensor: imgs})
        # print(tf_output)

        print('Running constant image through caffe net')
        reshaped_imgs = np.transpose(imgs, (0, 3, 1, 2))
        self.caffe_net.blobs['data'].data[...] = reshaped_imgs
        caf_output = self.caffe_net.forward()
        # print(caf_output)
        caf_output = caf_output['prob']

        assert(np.allclose(tf_output, caf_output))


if __name__ == '__main__':
    def_path = './data/ResNet-50-deploy.prototxt'
    caffemodel_path = './data/ResNet-50-model.caffemodel'
    data_path = './data/data.npy'

    conv_test = ConversionTest(def_path, caffemodel_path, data_path)
    conv_test.test_conversion()
    conv_test.close()
