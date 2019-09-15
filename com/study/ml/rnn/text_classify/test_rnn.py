'''
lstm:
    embedding
    lstm
    fc
    train_op
数据封装模块
    next_batch
词表封装：
    setense2id(text_setnece)
类别封装：
    category2id
'''

num_embedding_size = 16 #词向量长度
num_timesteps = 50, #可以是变长
num_lstm_nodes = [32,32]
num_lstm_layers = 2
num_fc_nodes = 32
batch_size = 100
clip_lstm_grads = 1.0 #梯度爆炸,消失不改结构如果解决


seg_train_file = "/Users/piguanghua/Downloads/cnews/cnews.train.seg.txt"
seg_val_file = "/Users/piguanghua/Downloads/cnews/cnews.train.val.txt"
seg_test_file = "/Users/piguanghua/Downloads/cnews/cnews.train.test.txt"
vocab_file = "/Users/piguanghua/Downloads/cnews/cnews.train.vocab.txt"
category_file = "/Users/piguanghua/Downloads/cnews/category.txt"

class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.split('\t')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx

