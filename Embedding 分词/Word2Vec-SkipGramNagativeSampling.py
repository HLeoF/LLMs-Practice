import codecs
from collections import Counter
from Skip_gram import SkipGramWithNegativeSampling
import numpy as np
from tqdm import tqdm


def read_and_build(file_path, window_size):
    """读取语料，构建词表等"""

    corpus = []
    with codecs.open(file_path, mode='r', encoding='utf-8') as fin:
        for line in fin:
            corpus.append(line.strip().split())

    # 计算词频
    word_counter = Counter([w for s in corpus for w in s])
    word_counts = word_counter.most_common()

    # 词表大小
    vocab_size = len(word_counts)

    # word to id
    word2id = {}
    # 词频归一化
    # unigram_dist = np.zeros(shape=(vocab_size,))
    # 总词数
    # total_words = sum([e[1] for e in word_counts])
    for idx, item in enumerate(word_counts):
        word2id[item[0]] = idx
    #     unigram_dist[idx] = item[1] / total_words

    # id to word
    id2word = dict(zip(word2id.values(), word2id.keys()))

    # 构建训练数据集：(中心词，context 词)
    training_data = []
    for s in corpus:
        for idx, w in enumerate(s):
            context = []
            for j in range(idx - window_size, idx + window_size + 1):
                if 0 <= j < len(s) and j != idx:
                    context.append(s[j])
            training_data.append((w, context))

    print(f'语料样本数: {len(corpus)}, 词表大小: {vocab_size}, 训练数据集大小：{len(training_data)}')

    return training_data, word2id, id2word, vocab_size, word_counts


def train(epochs, word_vec_file, pkl_file):

    for e in tqdm(range(epochs)):
        epoch_loss = 0.0

        for idx, sample in enumerate(training_data):

            # 前向和后向
            loss = sg.forward_and_back_propagate(sample)

            epoch_loss += loss

        # epoch 结束测试效果
        for test_w in ['活动', '公益', '中国', '运动鞋']:
            sg.most_similar(test_w)

        # epoch 结束存储词向量
        sg.save_to_file(word_vec_file + '_' + str(e), pkl_file + '_' + str(e))

        print(f'Epoch: {e}, loss: {epoch_loss / len(training_data)}')


if __name__ == '__main__':

    # 语料路径
    input_file_path = 'Embedding 分词/data//sohu.data'
    # 上下文窗口
    window_size = 2
    # 词向量大小
    embedding_size = 30
    # 学习率
    learning_rate = 0.01
    # epoch 数
    epoch_num = 30
    # 负样本数
    neg_count = 3

    training_data, word2id, id2word, vocab_size, word_counts = read_and_build(input_file_path, window_size)

    # 模型
    sg = SkipGramWithNegativeSampling(vocab_size, embedding_size, learning_rate, word_counts, neg_count, word2id, id2word)

    # 存储中间文件
    word_vec_file = 'Embedding 分词/data/word_vectors_ns'
    pkl_file = 'Embedding 分词/data/V_U_ns_pkl.pkl'

    train(epoch_num, word_vec_file, pkl_file)