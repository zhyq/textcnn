import argparse
from  data_handle import *
import tensorflow as tf
import logging
import codecs
from cnn import *
import os
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("title_len", 16,"Title Len (default:16)")
tf.flags.DEFINE_integer("content_len", 256,"Content Len (default:64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

batch_size = 64
title_len=16
content_len = 256
checkpoint_dir='model/'



def predict(args):
    logging.info("predict")
    # 训练和预测的预料
    corpus = os.path.join(args.path,args.corpus)
    # word 和 id的相互映射关系文件
    vocab_file = os.path.join(args.path,args.vocab)
    # 预测输出结果
    out_file = os.path.join(args.path,args.output)
    model_path = os.path.join(args.path,args.model)
    checkpoint_file = tf.train.latest_checkpoint(model_path)

    # 2 建立 词 和 id的映射关系 vocab
    logging.info("load vocab")
    vb = Vocab()

    vb.load(vocab_file)
    vb_size = len(vb.id_to_word)

    # 3 建立 array 和  str label对应关系
    lb = Label()

    # 4 初始化 model 模型
    logging.info("init model")

    logging.info("corpus: %s" % corpus)
    logging.info("vocab file: %s" % vocab_file)
    logging.info("vocab_size: %d" % vb_size)

    # 5 restore
    logging.info("restore checkpoint")
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session = tf.Session(config=session_conf)
        with session.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(session, checkpoint_file)
            input_title = graph.get_operation_by_name("input/input_title").outputs[0]
            input_content = graph.get_operation_by_name("input/input_content").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("input/dropout_keep_prob").outputs[0]
        
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            #ds = DataShuffle(words,vocab_file,title_len=title_len,content_len=content_len)
            ds = DataShuffle(corpus,vocab_file,title_len=title_len,content_len=content_len,load_file=False)
            out = codecs.open(out_file,'w','utf8')
            while True:
                title,content,corpus = ds.get_test_data()
                if 0 == title.shape[0] :
                    break
                feed_dict = {        
                    input_title : title,
                    input_content : content,
                    dropout_keep_prob : 1.0
                }
                preds =  session.run([predictions],feed_dict=feed_dict)
                pred = preds[0]
                str_label = lb.label_to_str(pred)
                out.write("%s\t%s\n" % (str_label,corpus))
            out.close()

def train(args):
    logging.info("train")
    # 训练和预测的预料
    corpus = os.path.join(args.path,args.corpus)
    # word 和 id的相互映射关系文件
    vocab_file = os.path.join(args.path,args.vocab)
    # 预先训练好的词向量模型 使用预先训练的词向量的好处 1 防止train预料不足导致预测 oov 2 词向量初始化更加科学
    w2v_model = args.word2vec
    model_path = os.path.join(args.path,args.model)
    logs_dir = os.path.join(args.path,args.logs)
    # 1 加载 ted 的xml 数据 data
    words = codecs.open(corpus,'r','utf8').read()
    # 2 建立 词 和 id的映射关系 vocab
    logging.info("build vocab")
    vb = Vocab()
    vb.build_vocab(words)
    vb.save(vocab_file)
    #emb_dim = 300
    vb_size = len(vb.id_to_word)
    #vb.load(vocab_file)
    # 3 加载预先训练的词向量 word embedding
    logging.info("load w2v model")
    vb_size,emb_dim,embd = load_pretrained_wv(w2v_model,vocab_file)
     
    logging.info("corpus: %s" % corpus)
    logging.info("vocab file: %s" % vocab_file)
    logging.info("word2vec model: %s" % w2v_model)
    logging.info("batch_size: %d" % batch_size)
    logging.info("title_len: %d" % title_len)
    logging.info("content_len: %d" % content_len)
    logging.info("vocab_size: %d" % vb_size)
    logging.info("emb_size: %d" % emb_dim)

    # 4 初始化 model 模型
    logging.info("init model")
    tc = TextClass(emb_dim=emb_dim, vocab_size=vb_size, title_len=title_len,content_len=content_len, class_num=3)

    # 5 session
    logging.info("init session")
    session = tf.Session()
    # tensorboard
    writer=tf.summary.FileWriter(logs_dir,session.graph)

    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=3)  # 最多保存的模型数量
    
    # 6 把 word2vec 模型加载到 model的emb中
    #"""
    logging.info("init model emb")
    feed_dict = {
        tc.emb_input : embd
    }
    session.run([tc.emb_init],feed_dict = feed_dict)
    #"""
    ds = DataShuffle(corpus,vocab_file,title_len=title_len,content_len=content_len,load_file=True)
    for step in range(500000):
        title,content,y =  ds.get_batch_data(batch_size=batch_size)

        feed_dict = {
                tc.input_title : title,
                tc.input_content : content,
                tc.input_y : y,
                tc.lr : 1e-3,
                tc.dropout_keep_prob : 0.5,
                }

        g_step,summary,train_step,acc = session.run([tc.global_step,tc.merged,tc.train_step,tc.accuracy],feed_dict=feed_dict)
        writer.add_summary(summary,g_step)
        if g_step % 100:
            logging.info("step %d,acc:%.3lf" % (g_step,acc))
        if step % 5000 == 0:
            save_path = saver.save(session, model_path, global_step=g_step)
            logging.info("save model to %s" % save_path)

    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "ted class")
    parser.add_argument("-a","--action",type=str,default="train",help="action train predict logs")
    parser.add_argument("-p","--path",type=str,default=".",help="base path")
    parser.add_argument("-o","--output",type=str,default=".",help="predict output path")
    parser.add_argument("-c","--corpus",type=str,default="data/train.txt",help="train or predict path")
    parser.add_argument("-m","--model",type=str,default="model/model.bin",help="model file")
    parser.add_argument("-v","--vocab",type=str,default="data/vocab.pkl",help="vocab file")
    parser.add_argument("-w","--word2vec",type=str,default="w2vmodel/wxp_300_w2v.txt",help="pretrained word2vec path")
    parser.add_argument("-l","--logs",type=str,default="logs",help="tensorboard logs")
    args = parser.parse_args()

    if args.action == "train":
        train(args)
    else :
        predict(args)
