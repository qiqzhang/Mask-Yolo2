import tensorflow as tf
import numpy as np

class BaseNetwork(object):
    def __init__(self,config, dropout = 0.5):

        self.config = config

        if self.config.pre_trained_npy_path is not None:
            self.data_dict = np.load(self.config.pre_trained_npy_path,encoding='latin1').item()
        else :
            self.data_dict =None


        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

       # self.build_network()

    def batch_norm(x, n_out, phase_train):
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def conv_layer(self , bottom, in_channels ,out_channels ,filter_size = 3,stride=1,name=None):
        with tf.variable_scope(name):
            filters,biases = self.get_conv_var(filter_size,in_channels,out_channels,name=name)
            bottom = tf.nn.conv2d(bottom, filters, strides=[1, stride, stride, 1], padding='SAME')
            bottom = tf.nn.bias_add(bottom, biases)
            return tf.nn.leaky_relu(bottom,0.1)

    def max_pool(self ,bottom,kernel_size,stride,name):
        return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1],padding='SAME',name=name)

    def fc_layer(self,bottom ,in_size ,out_size ,name):
        with tf.variable_scope(name):
            weights , biases = self.get_fc_var(in_size,out_size,name)
            x = tf.reshape(bottom,[-1,in_size])
            fc = tf.nn.bias_add(tf.matmul(x,weights),biases)
            return fc

    def get_conv_var(self,filter_size ,in_channels, out_channels ,name):
        init_val = tf.truncated_normal([filter_size,filter_size,in_channels,out_channels],0.0,0.001)
        filters = self.get_var(init_val,name,0,name+'_filters')

        init_val = tf.truncated_normal([out_channels],.0,.001)
        biases = self.get_var(init_val,name,1,name+'_biases')
        return filters,biases

    def get_fc_var(self,in_size,out_size ,name):
        init_val = tf.truncated_normal([in_size,out_size],.0,.001)
        weights = self.get_var(init_val,name,0,name+'_weights')

        init_val = tf.truncated_normal([out_size],.0,.001)
        biases = self.get_var(init_val,name,1,name+'_biases')
        return weights,biases

    def get_var(self,init_value,name,idx,var_name):
        #如果不用预训练数据则用随机初始化数据
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = init_value
        if self.config.trainable:
            var = tf.Variable(value,name=var_name)
        else:
            var = tf.constant(value,dtype=tf.float32,name=var_name)
        assert var.get_shape() == init_value.get_shape()
        return var

    def route_reorg_layer(self, input, in_channels ,out_channels ,depth,filter_size = 3,stride=1,name=None):
        with tf.variable_scope(name):
            output=self.conv_layer(input,in_channels ,out_channels,filter_size ,stride,name=name)
            output = tf.space_to_depth(output, depth)
            #  y = tf.concat([a, b], axis=3)
            return output

    def route_layer(self,layer1,layer2,name=None):
        with tf.variable_scope(name):
            return tf.concat([layer1,layer2],axis=3)
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def build_network(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

