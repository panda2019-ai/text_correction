# coding:utf-8
"""

"""
import os
from collections import Counter
import re
from pyhanlp import *

CorpusLoader = SafeJClass('com.hankcs.hanlp.corpus.document.CorpusLoader')
NatureDictionaryMaker = SafeJClass('com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker')
import time



# 统计语料语言学特征
def count_corpus(train_path: str):
    train_counter, train_freq, train_chars = count_word_freq(train_path)
    return train_chars / 10000, len(
        train_counter) / 10000, train_freq / 10000, train_chars / train_freq


# 返回词频词典f，词总数，总字符个数（总词长）
def count_word_freq(train_path):
    f = Counter()
    with open(train_path, errors='ignore') as src:
        for line in src:
            for word in re.compile("\\s+").split(line.strip()):
                f[word] += 1
    return f, sum(f.values()), sum(len(w) * f[w] for w in f.keys())


# 加载语料，并完成分句和分词，要求语料必须以空格分好词
def load_corpus(corpus_path):
    return CorpusLoader.convert2SentenceList(corpus_path)


# 训练2-gram模型
def train_bigram(corpus_path, model_path):
    t0 = time.time()
    sents = CorpusLoader.convert2SentenceList(corpus_path)
    for sent in sents:
        for word in sent:
            if word.label is None:
                word.setLabel("n")
    maker = NatureDictionaryMaker()
    maker.compute(sents)
    maker.saveTxtTo(model_path)  
    t1 = time.time()
    print("2-gram训练结束，时间=%.2f min"%((t1-t0)/60.0))


if __name__ == "__main__":
    # 设置数据名，这里用MSR语料或者用测试语料eng
    data_name = 'eng'
    # 设置训练集路径
    train_path = os.path.join('data/icwb2-data', 'training', '{}_training.utf8'.format(data_name))
    print("训练集路径：%s" % train_path)
    # 统计并输出语料语言学特征
    print('|语料库|字符数|词语种数|总词频|平均词长|')
    print('|%s|%.0f万|%.0f万|%.0f万|%.1f|' % ((data_name.upper(),) + count_corpus(train_path)))
    # # 加载语料并完成分句和分词
    # sents = list(load_corpus(train_path))
    # for sen in sents:
    #     print(type(sen))
    #     print(sen)
    # 设置模型保存路径
    model_path = os.path.join('data', '%s_correction_model' % data_name)
    print("2-gram模型路径：%s" % model_path)
    # 训练2-gram模型
    train_bigram(train_path, model_path)
    # 
    
    
