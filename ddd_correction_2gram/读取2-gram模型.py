# coding:utf-8
"""
"""
import os
from pyhanlp import *

CoreDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreDictionary')
CoreBiGramTableDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary')


if __name__ == "__main__":
    # 设置模型保存路径
    model_path = os.path.join('data', 'eng_correction_model')
    HanLP.Config.CoreDictionaryPath = model_path + ".txt"  # unigram
    HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  # bigram
    print("HanLP.Config.CoreDictionaryPath=%s" % HanLP.Config.CoreDictionaryPath)
    print("HanLP.Config.BiGramDictionaryPath=%s" % HanLP.Config.BiGramDictionaryPath)
    
    print(CoreDictionary.getTermFrequency("始##始"))
    print(CoreBiGramTableDictionary.getBiFrequency("始##始", "cats"))  # 始##始@cats

