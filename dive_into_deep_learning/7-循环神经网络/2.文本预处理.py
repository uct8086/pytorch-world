import collections
import re
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import read_time_machine, tokenize, Vocab, load_corpus_time_machine



 


if __name__ == '__main__':

    # lines = read_time_machine() 
    # print(f'# 文本总行数: {len(lines)}') 
    # print(lines[0])
    # print(lines[10])

    # tokens = tokenize(lines)
    # for i in range(11):
    #     print(tokens[i])

    # vocab = Vocab(tokens)
    # print(list(vocab.token_to_idx.items())[:10])

    # for i in [0, 10]:
    #     print('文本:', tokens[i]) 
    #     print('索引:', vocab[tokens[i]])


    corpus, vocab = load_corpus_time_machine()
    print(len(corpus), len(vocab))