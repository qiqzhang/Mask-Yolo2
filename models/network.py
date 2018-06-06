from base.base_network import *
import time

class NetWork(BaseNetwork):
  #  def __init__(self):
   #     super(NetWork,self).__init__()
       # self.build_network()

    def build_network(self , train_mode=None):

        start_time = time.time()
        print("build model started")

       # self.trainable = tf.placeholder(tf.bool)
        self.inputs = tf.placeholder(tf.float32,[None,224,224,3])
        self.ground_truth = tf.placeholder(tf.float32,[None,1000])

        self.conv1_1 = self.conv_layer(self.inputs, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.config.dropout), lambda: self.relu6)
        elif self.config.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.config.dropout)

        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.config.dropout), lambda: self.relu7)
        elif self.config.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.config.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.ground_truth,logits=self.prob))
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)

            correct_prediction = tf.equal(tf.argmax(self.prob,1) , tf.argmax(self.ground_truth , 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction ,tf.float32))

        print(("build model finished: %ds" % (time.time() - start_time)))

class letnet5(BaseNetwork):

    def build_network(self):

        start_time = time.time()
        print("build model started")




        print(("build model finished: %ds" % (time.time() - start_time)))

class YOLO2(BaseNetwork):
    def __init__(self, config):
        super(YOLO2, self).__init__(config)
        self.build_network()
        self.init_saver()

    def slice_tensor(self,x, start, end=None):

        #if end < 0:
        #    y = x[..., start:]

       # else:
         #   if end is None:
         #       end = start
         #   y = x[..., start:end + 1]
        if end is None:
            y = x[..., start:]
        else:
            y = x[..., start:end + 1]

        return y
    def build_network(self):

        self.istraining = tf.placeholder(tf.bool)

        with tf.name_scope("net"):
            start_time = int(round(time.time() * 1000))
            print("build yolo2 model started")

            self.inputs = tf.placeholder(tf.float32,[None,self.config.net_size_w,self.config.net_size_h,3])
            self.conv1 = self.conv_layer(self.inputs,3,32,filter_size=3,stride=1,name='conv1')
            self.maxpool1 = self.max_pool(self.conv1,2,2,name='maxpool1')

            self.conv2 = self.conv_layer(self.maxpool1,32,64,filter_size=3,stride=1,name='conv2')
            self.maxpool2 = self.max_pool(self.conv2, 2, 2, name='maxpool2')

            self.conv3 = self.conv_layer(self.maxpool2, 64, 128, filter_size=3, stride=1, name='conv3')
            self.conv4 = self.conv_layer(self.conv3, 128, 64, filter_size=1, stride=1, name='conv4')
            self.conv5 = self.conv_layer(self.conv4, 64, 128, filter_size=3, stride=1, name='conv5')
            self.maxpool3 = self.max_pool(self.conv5, 2, 2, name='maxpool3')

            self.conv6 = self.conv_layer(self.maxpool3, 128, 256, filter_size=3, stride=1, name='conv6')
            self.conv7 = self.conv_layer(self.conv6, 256, 128, filter_size=1, stride=1, name='conv7')
            self.conv8 = self.conv_layer(self.conv7, 128, 256, filter_size=3, stride=1, name='conv8')
            self.maxpool4 = self.max_pool(self.conv8, 2, 2, name='maxpool4')

            self.conv9 = self.conv_layer(self.maxpool4, 256, 512, filter_size=3, stride=1, name='conv9')
            self.conv10 = self.conv_layer(self.conv9, 512, 256, filter_size=1, stride=1, name='conv10')
            self.conv11 = self.conv_layer(self.conv10, 256, 512, filter_size=3, stride=1, name='conv11')
            self.conv12 = self.conv_layer(self.conv11, 512, 256, filter_size=1, stride=1, name='conv12')
            self.conv13 = self.conv_layer(self.conv12, 256, 512, filter_size=3, stride=1, name='conv13')
            self.maxpool5 = self.max_pool(self.conv13, 2, 2, name='maxpool5')

            self.conv14 = self.conv_layer(self.maxpool5, 512, 1024, filter_size=3, stride=1, name='conv14')
            self.conv15 = self.conv_layer(self.conv14, 1024, 512, filter_size=1, stride=1, name='conv15')
            self.conv16 = self.conv_layer(self.conv15, 512, 1024, filter_size=3, stride=1, name='conv16')
            self.conv17 = self.conv_layer(self.conv16, 1024, 512, filter_size=1, stride=1, name='conv17')
            self.conv18 = self.conv_layer(self.conv17, 512, 1024, filter_size=3, stride=1, name='conv18')
            self.conv19 = self.conv_layer(self.conv18, 1024, 1024, filter_size=3, stride=1, name='conv19')
            self.conv20 = self.conv_layer(self.conv19, 1024, 1024, filter_size=3, stride=1, name='conv20')

            self.reorg = self.route_reorg_layer(self.conv13,512,64,depth=2,filter_size=1,stride=1,name='reorg')
            # darknet 416x416 number 28
            self.route28 = self.route_layer(self.reorg,self.conv20,name='route28')
            self.conv21 = self.conv_layer(self.route28, 1280, 1024, filter_size=3, stride=1, name='conv21')
            self.conv22 = self.conv_layer(self.conv21,1024,self.config.num*(5+self.config.classes),filter_size=1,stride=1,name='conv22')

            self.output = tf.reshape(self.conv22,shape=(-1, self.config.grid, self.config.grid, self.config.num, 5+self.config.classes ) ,name='output')
            print(("build yolo2 model finished: %d ms" % (int(round(time.time() * 1000)) - start_time)))


        with tf.name_scope("loss"):
            start_time = int(round(time.time() * 1000))
            print("build loss started")
            self.ground_truth = tf.placeholder(shape=(None, self.config.grid, self.config.grid, self.config.num,6),dtype=tf.float32)
            mask = self.slice_tensor(self.ground_truth, 5)
            label = self.slice_tensor(self.ground_truth, 0, 4)
           # print(mask)
            mask = tf.cast(tf.reshape(mask, shape=(-1, self.config.grid, self.config.grid, self.config.num)), tf.bool)

            with tf.name_scope('mask'):
                masked_label = tf.boolean_mask(label, mask)
                masked_pred = tf.boolean_mask(self.output, mask)
             #   print(masked_pred)
                neg_masked_pred = tf.boolean_mask(self.output, tf.logical_not(mask))

            with tf.name_scope('pred'):
                masked_pred_xy = tf.sigmoid(self.slice_tensor(masked_pred, 0, 1))

               # masked_pred_wh = tf.exp(self.slice_tensor(masked_pred, 2, 3))
                masked_pred_wh = self.slice_tensor(masked_pred, 2, 3)
                masked_pred_o = tf.sigmoid(self.slice_tensor(masked_pred, 4,4))
                masked_pred_no_o = tf.sigmoid(self.slice_tensor(neg_masked_pred, 4,4))
                masked_pred_c = tf.nn.softmax(self.slice_tensor(masked_pred, 5))
               # print(masked_pred_c)
            with tf.name_scope('lab'):
                masked_label_xy = self.slice_tensor(masked_label, 0, 1)
                masked_label_wh = self.slice_tensor(masked_label, 2, 3)
                masked_label_c = self.slice_tensor(masked_label, 4,4)
               # print(masked_label_c)
                masked_label_c_vec = tf.reshape(tf.one_hot(tf.cast(masked_label_c, tf.int32), depth=self.config.classes),
                                                shape=(-1, self.config.classes))

            with tf.name_scope('merge'):
                with tf.name_scope('loss_xy'):
                    self.loss_xy = tf.reduce_sum(tf.square(masked_pred_xy - masked_label_xy))

                with tf.name_scope('loss_wh'):
                    self.loss_wh = tf.reduce_sum(tf.square(masked_pred_wh -masked_label_wh))

                with tf.name_scope('loss_obj'):
                    self.loss_obj = tf.reduce_sum(tf.square(masked_pred_o - 1))

                with tf.name_scope('loss_no_obj'):
                    self.loss_no_obj = tf.reduce_sum(tf.square(masked_pred_no_o))

                with tf.name_scope('loss_class'):
                    #self.loss_c = tf.reduce_sum(tf.square(masked_pred_c - masked_label_c_vec))
                    self.loss_c = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=masked_label_c_vec,logits=masked_pred_c))

#                print("loss_xy:%f,loss_wh:%f,loss_obj:%f,loss_no_obj:%f,loss_class:%f"%(loss_xy,loss_wh,loss_obj,loss_no_obj,loss_c))
                self.loss = self.config.lambda_coord * (self.loss_xy + self.loss_wh) + self.loss_obj + self.config.lambda_no_obj * self.loss_no_obj + self.loss_c
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                             global_step=self.global_step_tensor)
            print(("build loss finished: %d ms" % (int(round(time.time() * 1000)) - start_time)))

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

















































