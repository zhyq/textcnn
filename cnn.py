import tensorflow as tf
import sklearn
from functools import reduce
class TextClass():
    def __init__(self,emb_dim,vocab_size,title_len,content_len,class_num):
        """

        """
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.title_len = title_len
        self.content_len = content_len
        self.class_num = class_num
        self.global_step = tf.Variable(0,trainable=False,name="global_step")
        with tf.name_scope("input"):
            self.input_title = tf.placeholder(tf.int32,[None,self.title_len],name="input_title")
            self.input_content = tf.placeholder(tf.int32,[None,self.content_len],name="input_content")
            self.input_y = tf.placeholder(tf.int32,[None,self.class_num],name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
            self.lr = tf.placeholder(tf.float32, [])


        with tf.name_scope('layer'):
            self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size,self.emb_dim],minval=-1.0,maxval=1.0,dtype=tf.float32),trainable=True)
            #self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size,self.emb_dim],minval=-1.0,maxval=1.0,dtype=tf.float32),trainable=False)
            self.emb_input = tf.placeholder(tf.float32,[self.vocab_size,self.emb_dim],name="emb_input")
            self.emb_init = self.emb.assign(self.emb_input) 
            self.title_emb = tf.nn.embedding_lookup(self.emb,self.input_title)
            self.content_emb = tf.nn.embedding_lookup(self.emb,self.input_content)

            l2_loss = tf.constant(0.0)
            
            # cnn
            #卷积和高度 为 3 4 5 ，对于文本卷积，卷积的宽度为emb_dim
            filter_sizes = [3,4,5]
            #卷积的通道数 决定这卷积的输出大小
            num_filters = [128,128,128]
            num_filter_total = reduce(lambda x,y:x+y,num_filters)
            self.title = self.cnn(self.title_emb,self.title_len,filter_sizes,num_filters,num_filter_total)
            self.content = self.cnn(self.content_emb,self.content_len,filter_sizes,num_filters,num_filter_total)
            self.title_content_concat = tf.concat([self.title,self.content],axis=1,name='concat')
            #self.input_x = tf.concat([self.title_emb,self.content_emb],axis = 1)
            #self.title_content_concat = self.cnn(self.input_x,title_len+content_len,filter_sizes,num_filters,num_filter_total)
            self.x = tf.nn.dropout(self.title_content_concat,self.dropout_keep_prob)
            self.W = tf.Variable(tf.random_uniform([2*num_filter_total,self.class_num],-1.0,1.0),name='W')
            self.b = tf.Variable(tf.zeros(self.class_num),name='b')
            #self.y = tf.matmul(self.x,self.W) + self.b
            l2_loss += tf.nn.l2_loss(self.W)
            l2_loss += tf.nn.l2_loss(self.b)

            self.y = tf.nn.xw_plus_b(self.x,self.W,self.b,name='score')

            tf.summary.histogram('layer/weights', self.W)
            tf.summary.histogram('layer/bias', self.b)
            tf.summary.histogram('layer/output', self.y)
        with tf.name_scope('output'):
            self.predictions = tf.argmax(self.y,1,name='predictions')
        with tf.name_scope('optimizer'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.input_y)
            l2_reg_lambda = 0.0
            self.cross_entropy = tf.reduce_mean(losses,name='loss') + l2_reg_lambda * l2_loss
            
            #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cross_entropy)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads_and_vars = optimizer.compute_gradients(self.cross_entropy)
            self.train_step = optimizer.apply_gradients(grads_and_vars,global_step=self.global_step)

        with tf.name_scope('loss'):
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32),name='accuracy')
            tf.summary.scalar('acc', self.accuracy)
            self.merged = tf.summary.merge_all()

    def cnn(self,x,sentence_len,filter_sizes,num_filters,num_filter_total):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        def max_pool(x,filter_size):
            return tf.nn.max_pool(x, ksize=[1, sentence_len - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID')

        #x_image = tf.reshape(x,[-1,sentence_len,self.emb_dim,1])
        x_image = tf.expand_dims(x,-1)

        pooled_outputs = []

        for filter_size,num_filter in zip(filter_sizes,num_filters):
            W_conv = weight_variable([filter_size, self.emb_dim, 1, num_filter])
            b_conv = bias_variable([num_filter])
            h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
            h_pool = max_pool(h_conv,filter_size)
            pooled_outputs.append(h_pool)

        h_pool = tf.concat(pooled_outputs,axis=3)
        h_pool_flat = tf.reshape(h_pool,[-1,num_filter_total])
        #h_pool_dropout = tf.nn.dropout(h_pool_flat,self.dropout_keep_prob)
        #return h_pool_dropout
        return h_pool_flat
