import codecs
import numpy as np
from tqdm import tqdm
from Skip_gram import SkipGram
from collections import Counter

def read_and_build(file_path, window_size):
    """读取语料，构建词表"""
    corpus = []
    with codecs.open(file_path, mode='r', encoding='utf-8') as fin:
        for line in fin:
            corpus.append(line.strip().split())
    
    #计算词频
    word_counter = Counter([w for s in corpus for w in s])
    word_counts = word_counter.most_common()

    #词表大小
    vocab_size = len(word_counts)

    #Word to ID
    word2id = {}
    for idx, item in enumerate(word_counts):
        word2id[item[0]] = idx

    #ID to Word
    id2word = dict(zip(word2id.values(), word2id.keys()))

    # 构建训练数据集：（中心词，context词）
    training_data = []
    for s in corpus:
        for idx, w in enumerate(s):
            context = []
            for j in range(idx-window_size, idx+window_size + 1):
                if 0 <= j < len(s) and j != idx:
                    context.append(s[j])
            training_data.append((w,context))
    
    print(f'语料样本数：{len(corpus)}, 词表大小：{vocab_size}, 训练数据集大小：{len(training_data)}')

    return training_data, word2id, id2word, vocab_size


def train(epochs, word2id, word_vec_file, pkl_file):
    """模型训练"""
    
    for e in range(epochs):
        epoch_loss = 0.0

        for idx, sample in enumerate(training_data):
            w, context = sample

            w_onehot = np.zeros(shape=(vocab_size,1))
            w_onehot[word2id[w]][0] = 1

            h, y_pred, u  = sg.forward(w_onehot)

            sg.back_propagate(context, h, y_pred, w_onehot)

            # loss 汇总
            # see https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling#eq-7
            epoch_loss += -np.sum([u[word2id[c]]] for c in context) + len(context) * np.log(np.sum(np.exp(u)))

            for test_w in ['活动','公益','中国','运动鞋']:
                sg.most_similar(test_w)

            sg.save_to_file(word_vec_file + '_' + str(e), pkl_file + '_' + str(e))

            print(f'Epoch: {e}, loss: {epoch_loss/len(training_data)}')

if __name__ == "__main__":

    #语料路径
    file_path = "Embedding 分词/data/sohu.data"

    #上下文窗口
    window_size = 2

    #词向量大小
    embedding_size = 30

    #学习率
    learning_rate = 0.01

    #Epoch 数
    epoch_num = 1

    training_data, word2id, id2word, vocab_size=read_and_build(file_path,window_size)

    #Skip-Gram 模型
    sg = SkipGram(vocab_size, embedding_size, learning_rate, word2id, id2word)

    word_vec_file = "Embedding 分词/data/word_vectors"
    pkl_file = 'Embedding 分词/data/V_U_Pkl.pkl'

    train(epoch_num, word2id, word_vec_file, pkl_file)
