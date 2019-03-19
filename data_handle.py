import xml.etree.ElementTree as ET
import collections
import pickle
import numpy
import codecs
class Vocab():
    def __init__(self):
        self.id_to_word=[]
        self.word_to_id=[]

    def build_vocab(self,words):
        data = words.replace("\n"," </s> ").split()
        counter = collections.Counter(data)
        pair_data = sorted(counter.items(),key=lambda x:(-x[1],x[0]))
        pair_data = list(filter(lambda x:x[1]>2,pair_data))
        words,_ = list(zip(*pair_data))
        words = list(words)
        ### oov to zero
        words.insert(0,'<oov>')
        self.word_to_id = dict(zip(words,range(len(words))))
        self.id_to_word = dict(zip(range(len(words)),words))

    def word2id(self,words):
        ids = [self.word_to_id.get(word,0) for word in words.split()]
        return ids

    def id2word(self,ids):
        words = [self.id_to_word.get(id,'<oov>') for id in ids.split()]
        return " ".join(words)

    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.word_to_id,f)
            pickle.dump(self.id_to_word,f)

    def load(self,path):
        with open(path,'rb') as f:
            self.word_to_id = pickle.load(f)
            self.id_to_word = pickle.load(f)


def load_pretrained_wv(w2v_file,vocab_file):
    """
        input: w2v_file word2vec 模型
               vocab_file Vocab(word 和 id 对应关系的数据结构)加载后存储为pkl的文件
        return: vb_size:词典大小
                emb_size:词向量维度
                embd:array 加载的词向量 (词的vocab中id为array的小标)
    """
    vb = Vocab()
    vb.load(vocab_file)
    vb_size = len(vb.word_to_id.keys())

    f = codecs.open(w2v_file,'r','utf8')
    line = f.readline()
    _,emb_size = line.split(' ')
    emb_size = int(emb_size)

    embd = [[0]*emb_size]*vb_size
    for line in f.readlines():
        row = line.strip().split(' ')
        # oov
        idx=vb.word_to_id.get(row[0],emb_size)
        if idx != emb_size:
            embd[idx]=row[1:]
    print('Loaded word2vec!')
    f.close()
    return vb_size,emb_size,embd

class Label():
    def __init__(self):
        self.labels=['cell','makeup','neg']
        self.label_to_id = dict(zip(self.labels,range(len(self.labels))))
        self.id_to_label = dict(zip(range(len(self.labels)),self.labels)) 
    def str_to_label(self,strlabel):
        labels_id = self.label_to_id.get(strlabel,0)
        y=[0] * len(self.labels)
        y[labels_id] = 1
        return y    
    def label_to_str(self,label):
        ids = label[0]
        if ids > len(self.labels):
            return 'err'
        return self.id_to_label[ids]


class DataShuffle():
    def __init__(self,corpus_file,vocab_file,title_len=16,content_len=256,load_file=True):
        self.line_number = 0
        self.vb = Vocab()
        self.vb.load(vocab_file)
        self.lb = Label()
        self.title = []
        self.content = []
        self.y = []
        self.corpus = []
        self.f_corpus = codecs.open(corpus_file,'r','utf8')
        self.title_len = title_len
        self.content_len = content_len
        if load_file:
            corpus = self.f_corpus.read()
            lines = corpus.split('\n')  

            for line in lines:
                line_list = line.split('\t')
                if len(line_list) != 3 and len(line_list) != 2:
                    #print("line error %d %s" % (len(line_list),line))
                    print("line error %d" % len(line_list))
                    continue
                if len(line_list) == 3:
                    label,title,content = line_list
                if len(line_list) == 2:
                    label = 'cell'
                    title,content = line_list
                title_id = self.fill_cut(title,self.title_len)
                content_id = self.fill_cut(content,self.content_len)
                y = self.lb.str_to_label(label)
                self.title.append(title_id)
                self.content.append(content_id)
                #if len(line_list) == 2:
                self.corpus.append(line)
                #self.y.append(labels_id)
                self.y.append(y)

            assert len(self.title) == len(self.y) and len(self.content) == len(self.y)

            
    def fill_cut(self,string,ids_len):
        ids = self.vb.word2id(string)
        if len(ids) < ids_len:
            more_id = [0] * (ids_len - len(ids))
            ids.extend(more_id)
        else:
            ids = ids[:ids_len]
        return ids



    def get_batch_data(self,batch_size=128):
        random_index = numpy.random.choice(len(self.title), batch_size, replace=True)
        title = []
        content = []
        y = []
        for i in range(len(random_index)):
            title.append(self.title[random_index[i]])
            content.append(self.content[random_index[i]])
            y.append(self.y[random_index[i]])

        title = numpy.array(title)
        content = numpy.array(content)
        y = numpy.array(y)
        return title,content,y


    def get_test_data(self):
        if len(self.title) > 0:
            if self.line_number < len(self.title):
                title_id = self.title[self.line_number]
                content_id = self.content[self.line_number]
                ret_title = numpy.array(title_id).reshape([-1,len(title_id)])
                ret_content = numpy.array(content_id).reshape([-1,len(content_id)])
                try:
                    corpus = self.corpus[self.line_number]
                except:
                    corpus = ''
                self.line_number += 1
                return ret_title,ret_content,corpus
            else :
                return numpy.array([]),numpy.array([]),''
        else:
            while True:
                line = self.f_corpus.readline()
                if line == '':
                    return numpy.array([]),numpy.array([]),''
                line = line.strip()
                line_list = line.split('\t')
                if len(line_list) == 2 or len(line_list) == 3:
                    break
            title,content = line_list[-2:]
            title_id = self.fill_cut(title,self.title_len)
            content_id = self.fill_cut(content,self.content_len)
            ret_title = numpy.array(title_id).reshape([-1,len(title_id)])
            ret_content = numpy.array(content_id).reshape([-1,len(content_id)])
            return ret_title,ret_content,line




if __name__ == "__main__":
    print("hello")
    vb = Vocab()
    #vb.build_vocab(words)
    #print(len(vb.word_to_id))
    #vb.save("data/vocab.pkl")
    vb.load("data/vocab.pkl")
    print(vb.word_to_id.keys())
