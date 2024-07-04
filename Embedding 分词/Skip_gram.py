import math
import codecs
import pickle
import numpy as np
from scipy.spatial.distance import cdist

class SkipGram:

    #SkipGram类初始化
    def __init__(self, vocab_size, dim, lr, word2id, id2word):
        self.vocab_size = vocab_size #词汇表大小
        self.dim = dim               #词向量维度
        self.lr = lr                 #学习率
        self.word2id = word2id       #词到ID的映射
        self.id2word = id2word       #ID到词的映射

        #中心词的词向量矩阵 V, 大小为(vocab_size(行), dim(列))
        #uniform 制定范范围内均匀分布的输随机数
        self.V = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size,dim))

        #上下文词的词向量矩阵 U, 大小同样也是为(vocab_size(行)， dim(列))
        self.U = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(dim,vocab_size))

    #前向传播
    def forward(self, w):
        """
            向前传播，它将输入数据通过矩阵运算进行处理，从而得到输出预测。
            其主要的目标是计算模型的输出，从而在后续的步骤中可以评估预测结果于实际
            目标之间的差距（误差）

            输入层：输入层接收原始数据
            隐藏层：数据通过隐藏层处理，对数据进行加权求和，然后应用激活函数（ReLU, Sigmod等）
                   以非线性的方式转换输入（通常包含多层）
            输出层：最后一层对隐藏层的结果进行处理，得到最终的输出。
        """
        h = np.dot(self.V.T, w) # h 隐藏值的输出， 形状为（dim, 1）
        
        # u 输出层的输入:表示隐藏层输出 h 经过第二层神经元的线性组合
        # 形状为（vocab_size,1),
        u = np.dot(self.U.T, h) 

        #y是输出层的激活值，
        # 表示输入 w 对应的概率分布，形状为（vocab_size,1)
        y = self.softmax(u)  

        #squeeze() 目的是将矩阵形状（vocab_size, 1) 变成 (vocab_size,)
        return h, y.squeeze(), u.squeeze() 
    
    #反向传播
    def back_propagate(self, context, h, y_pred, x):
        """
         向后传播：其核心思想是通过计算损失函数相对于每个权重的梯度，并利用梯度重新更新这些权重
         从而最小化损失函数

         主要步骤：
         1. 前向传播
         2. 计算损失梯度Gradient of the Loss: 从输出层开始，计算损失函数相对于每个神经元输出的梯度
            使用Chain Rule 逐层向后计算梯度，直到输入层
         3. 更新权重：Weight Update
            使用梯度下降算法，根据计算得到梯度更新每层权重和偏置
        """
        #初始化上下文向量列表
        context_vecs = []
        for c in context:
            context_vec = np.zeros(shape=(self.vocab_size, ))
            context_vec[self.word2id[c]] = 1
            context_vecs.append
        
        #计算误差 ei, 队员每个上下文词语的孤热编码向量 c, 计算预测输出 y_pred 和 c 的差值
        #将这些差值向量相加，得到误差项 ei
        ei = np.sum([np.subtract(y_pred, c) for c in context_vecs], axis=0)

        #dl_du是损失函数对于权重矩阵 u 的梯度， 通过计算隐藏层输出 h 和 误差项 ei 的外积
        dl_du = np.outer(h,ei)
        #dl_dv 是损失函数对于权重 v 的梯度， 通过计算向量 x 和 U 与 ei.T 点积的外积得到的
        dl_dV = np.outer(x, np.dot(self.U, ei.T))

        #根据梯度更新权重矩阵 V 和 U. self.lr 是学习率，用于控制权重更新的步长
        self.V = self.V - self.lr * dl_dV
        self.U = self.U - self.lr * dl_du

    #Softmax 函数
    def softmax(self, x):
        """
        将输入向量转换为概率分布，输出每个类的概率
        """
        # 计算输入向量 x 的最大值，目的是为了数值的稳定性，防止该函数溢出，当输入数据非常大时，
        # 指数函数的值可能变得非常大，从而导致数值不稳定
        e_x = np.exp(x - np.max(x))
        #计算归一后的概率分布
        return e_x / e_x.sum(axis = 0)
    
    def most_similar(self, w, topk=10):
        # w 要查询的相似词语的输入词
        # topk: 返回相似度搞得前 topk 个词，默认为10
        
        # 获取输入词的词向量，将输入词 w 映射字典的索引，从词向量矩阵中获取对应的词向量 w_vec
        w_vec = self.V[self.word2id[w]]

        # 计算与输入词的余弦相似度。 计算输入词向量 w_vec 与 
        # 词向量矩阵 self.V 中所有词向量的余弦距离
        distance = cdist(np.array([w_vec]), self.V, "cosine")[0]

        tmp_sims = 1-distance
        #找到相似度最高的词语，tep_sims 相似度数组 按照升序进行排序，返回排序后的索引数组
        topk_similarity_idx = np.argsort(tmp_sims)[::-1][:topk]
        print(f"Top {topk} similar words for {w}")
        for sim in topk_similarity_idx:
            print(f"Word: {self.id2word[sim]}, similar value: {tmp_sims[sim]}")

    #保存模型
    def save_to_file(self, save_file, pkl_file):
        with codecs.open(save_file, mode="w", encoding='utf-8') as fout:
            fout.write(str(self.vocab_size) + '\t' + str(self.dim) + '\n')
            for w in range(self.V.shape[0]):
                fout.write(self.id2word[w] + '\t' + ' '.join([str(val) for val in self.V[w]]) + '\n')
            
        with open(pkl_file, 'wb') as fout:
            pickle.dump([self.V, self.U], fout)

    

class SkipGramWithNegativeSampling:
    """
    Skip-Gram 负采样用于Word2Vec模型中的一种训练方法， 负采样是Skip-Gram的一种优化方案，用于替代
    原始的Softmax计算，以加快Skip-Gram模型的训练速度和效率。 在传统的Skip-Gram模型中使用Softmax 计算
    词汇表中每个词作为上下文的概率，这在大型词汇表上计算成本很高。所有使用Negative Sampling 通过随机选择一小部分负例（
    既不是目标词的词）来近似softmax计算，从而减少计算量。

    Negative-Sampling的关键思想，对于给定的目标词和上下文词，我们随机选择一些词作为负例，而后只更新这些负例的向量
    和目标词的向量表示。这样就可以有效减少训练过程中计算复杂度，同时保持模型的效果。
    """
    def __init__(self,vocab_size, dim, lr, word_counts, neg_count,word2id, id2word):
        self.vocab_size = vocab_size #词汇表大小
        self.lr = lr                 #学习率
        self.dim = dim               #词向量维度
        self.neg_count = neg_count   # 负例词汇的数量
        self.word2id = word2id       #词到ID的映射
        self.id2word = id2word       #ID到词的映射

        self.table = self.init_table(word_counts) # word_counts -> 每个语料中出现的频率
                                                  #初始化负采样的概率表
        

        #中心词的词向量矩阵 V, 大小为(vocab_size(行), dim(列))
        #uniform 制定范范围内均匀分布的输随机数
        self.V = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size,dim))

        #上下文词的词向量矩阵 U, 大小同样也是为(vocab_size(行)， dim(列))
        self.U = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size,dim))  

    def init_table(self, word_counts):
        """
        通过调整和归一化词频来生成噪声分布概率表，用于负采样过程中随机选择
        """
        power = 0.75 #用来调整整词频权重参数，通常为“平滑参数”（Smoothing parameter）
        #计算所有词频 power 次幂的总和，得到归一因子norm
        norm = sum([math.pow(t[1],power) for t in word_counts]) 

        noise_dist_normalized = {} #初始化一个空字典 noise_dist_normalized, 用于存储归一化后的噪声分布概率
        for w, c in word_counts:
            noise_dist_normalized[self.word2id[w]] = float(math.pow(c,power))/norm
        print(f"Sum of Noise Dist Normalized Value : {sum(noise_dist_normalized.values())}")
        
        return noise_dist_normalized
        
    ## 负样本采集
    def negative_sample(self, count):
        """
        self.table.keys() 获取负采样概率表的所有ID
        self.table.values() 获取负采样概率表的所有值（每个表的采样概率）
        np.random.choice():
            list(self.table.keys()): 待选样本（词的ID）
            size: 采样数量
            p: 每个样本被选中的概率（从self.table.values()获取概率）
        """
        return list(np.random.choice(list(self.table.keys()), size=count, p=list(self.table.values())))

    #Sigmoid 激活函数
    def sigmoid(self, z):
        """它将任意实数输入映射到(0,1) 区间"""
        return 1.0 / (1.0 + np.exp(-z))
    
    #前向传播 & 后向传播
    def forward_and_back_propagate(self, sample):
        """
        通过Forward Propagate 计算样本损失，通过Back Propagate更新词向量。
        在每次迭代中生成负样本，从而提高训练效率，返回Loss用于监控模型训练过程
        """
        w_src, context_src = sample

        #将词转换成ID
        w = self.word2id[w_src]
        context = [self.word2id[c] for c in context_src]
        
        loss = 0.0 #初始化Loss

        #遍历每个上下文词
        for c in context:
            context_sum = np.zeros(self.dim)
            
            #生成正样本和负样本标签
            labels = [(c,1)] + [(neg, 0) for neg in self.negative_sample(self.neg_count)]

            #前向 & 后向
            for target,label in labels:
                #Forward Propagate
                z = np.dot(self.V[w], self.U[target])
                p = self.sigmoid(z)
                g = self.lr * (p-label)
                context_sum += g * self.U[target]
                #Back Propagate
                self.U[target] -= g*self.V[w]
            
            #更新中心词向量
            self.V[w] -= context_sum

            #计算Loss
            for target, label in labels:
                if label == 1:
                    loss += -np.log(self.sigmoid(np.dot(self.V[target], self.V[w])))
                else:
                    loss += -np.log(self.sigmoid(np.dot(-self.V[target], self.V[w])))
        
        return loss
    
    #找最相似的
    def most_similar(self, w, topk=10):
        w_vec = self.V[self.word2id[w]]

        distances = cdist(np.array([w_vec]), self.V, "cosine")[0]
        tmp_sims = 1 -distances

        topk_similarity_idx = np.argsort(tmp_sims)[::-1][:topk]

        print(f'Top {topk} similar words for: {w}')
        for sim in topk_similarity_idx:
            print(f'Word: {self.id2word[sim]}, similar value:{tmp_sims[sim]}')
    
    #模型存储
    def save_to_file(self, save_file, pkl_file):
        #codes 用于处理文件的编码问题，特别是非ASCII字符
        with codecs.open(save_file, mode='w', encoding='utf-8') as fout:
            #将词汇大小和词向量维度写入文件的第一行，以制表符分隔
            fout.write(str(self.vocab_size) + '\t' + str(self.dim) + '\n')

            # 对于每个词，将词的字符串表示和对应的词向量写入文件。词向量的每个元素用空格分隔，
            # 与词的字符串表示用制表符分隔
            for w in range(self.V.shape[0]):
                fout.write(self.id2word[w] + '\t' + ' '.join([str(val) for val in self.V[w]]) + '\n')
            
            #使用pickle 用于将Python对象序列化为二进制格式，以便后续反序列化加载
            with open(pkl_file,'wb') as fout:
                pickle.dump([self.V, self.U], fout)