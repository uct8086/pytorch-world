import torch
from torch import nn
import sys
sys.path.append("..")
print(sys.path)
from dive_into_deep_learning.d2l.d2l_torch import Seq2SeqEncoder, Decoder, try_gpu, load_data_nmt, EncoderDecoder, train_seq2seq, predict_seq2seq, bleu

class Seq2SeqDecoder(Decoder): 
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]
    
    def forward(self, X, state):
        # 输出'X'的形状:(batch_size,num_steps,embed_size) 
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1) 
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state) 
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size) 
        # # state的形状:(num_layers,batch_size,num_hiddens) 
        # return output, state
        return output, state

if __name__ == '__main__':

    # encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
    #                      num_layers=2)
    # encoder.eval()
    # X = torch.zeros((4, 7), dtype=torch.long)
    # output, state = encoder(X)
    # print(output.shape)


    # decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
    #                      num_layers=2)
    # decoder.eval()
    # state = decoder.init_state(encoder(X))
    # output, state = decoder(X, state)
    # print(output.shape, state.shape)


    # 训练

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, try_gpu()
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                            dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                            dropout)
    net = EncoderDecoder(encoder, decoder)
    # train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 预测

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')