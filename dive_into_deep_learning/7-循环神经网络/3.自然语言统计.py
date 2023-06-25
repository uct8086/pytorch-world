import random
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import read_time_machine, tokenize, Vocab, plot

if __name__ == '__main__':

    tokens = tokenize(read_time_machine())
    # 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起 
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    # print(vocab.token_freqs[:10])

    freqs = [freq for token, freq in vocab.token_freqs]
    # plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
    #      xscale='log', yscale='log')
    
    # 二元语法
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    # 三元语法

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    # 三者对比
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
            ylabel='frequency: n(x)', xscale='log', yscale='log',
            legend=['unigram', 'bigram', 'trigram'])