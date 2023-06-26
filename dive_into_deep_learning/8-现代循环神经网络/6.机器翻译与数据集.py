import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import read_data_nmt, preprocess_nmt, tokenize_nmt, show_list_len_pair_hist, Vocab, truncate_pad, load_data_nmt


if __name__ == '__main__':

    raw_text = read_data_nmt()
    # print(raw_text[:75])

    text = preprocess_nmt(raw_text)
    print(text[:80])

    source, target = tokenize_nmt(text)
    # print(source[:6], target[:6])

    # show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
    #                     'count', source, target)
    
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(len(src_vocab))

    print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

    # 下面我们读出“英语-法语”数据集中的第一个小批量数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32)) 
        print('X的有效⻓度:', X_valid_len) 
        print('Y:', Y.type(torch.int32)) 
        print('Y的有效⻓度:', Y_valid_len) 
        break