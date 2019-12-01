# coding:utf-8
"""
基于n-gram模型的，地，得语法纠正
"""

import os
from pyhanlp import *
import codecs
from nltk import ngrams
import re
import numpy as np

CoreDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreDictionary')
CoreBiGramTableDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary')

# 设置模型保存路径
model_path = os.path.join('data', 'MSR_correction_model')
HanLP.Config.CoreDictionaryPath = model_path + ".txt"  # unigram
HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  # bigram
print("HanLP.Config.CoreDictionaryPath=%s" % HanLP.Config.CoreDictionaryPath)
print("HanLP.Config.BiGramDictionaryPath=%s" % HanLP.Config.BiGramDictionaryPath)


# 计算pre_word@de@pro_word代价
def caculate_weight(pre_word, de_word, pro_word):
    # 计算pre_word@de的代价
    if CoreDictionary.getTermFrequency(pre_word) != 0:
        pre_weight = CoreBiGramTableDictionary.getBiFrequency(pre_word, de_word)/CoreDictionary.getTermFrequency(pre_word)
    else:
        pre_weight = 0

    # 计算pro_word@de的代价
    if CoreDictionary.getTermFrequency(de_word):
        pro_weight = CoreBiGramTableDictionary.getBiFrequency(de_word, pro_word)/CoreDictionary.getTermFrequency(de_word)
    else:
        pro_weight = 0
    
    # 计算 pre_word@de@pro_word代价
    cost = (pre_weight + pro_weight) / 2

    return cost

# 

if __name__ == "__main__":
    candidate_word_li = [u'的', u'地', u'得']
    with codecs.open('data/test/的地得词组练习.txt', 'rb', 'utf-8', 'ignore') as infile:
        for line in infile:
            line = line.strip()
            if line:
                pre_word, cur_word, pro_word = re.split(u'【|】', line)
                candidate_weight_li = []
                for candidate_word in candidate_word_li:
                    candidate_weight_li.append(caculate_weight(pre_word, candidate_word, pro_word))
                predict_word = candidate_word_li[np.argmax(candidate_weight_li)]
                print(predict_word, line, candidate_weight_li)




