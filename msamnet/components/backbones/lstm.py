import torch
import torch.nn as nn
from seqtr.models import LAN_ENCODERS
import numpy as np
from torch.autograd import Variable
def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, maxtoken=15, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        # new_global = l2norm(new_global, dim=-1)

        return new_global


@LAN_ENCODERS.register_module()
class LSTM(nn.Module):
    def __init__(self,
                 num_token,
                 word_emb,
                 lstm_cfg=dict(type='gru',
                               num_layers=1,
                               dropout=0.,
                               hidden_size=512,
                               bias=True,
                               bidirectional=True,
                               batch_first=True),
                 output_cfg=dict(type="max"),
                 freeze_emb=True):
        super(LSTM, self).__init__()
        self.fp16_enabled = False
        self.num_token = num_token

        assert len(word_emb) > 0
        lstm_input_ch = word_emb.shape[-1]
        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(word_emb),
            freeze=freeze_emb,
        )# 该处的embedding使用了glove模型的提取向量，单个单词为300维
        # 该embedding层也不需要训练，freeze=True,参数直接为 Embedding(10344, 300)
        assert lstm_cfg.pop('type') in ['gru']
        self.lstm = nn.GRU(**lstm_cfg, input_size=lstm_input_ch) # bi-gru used
        
        output_type = output_cfg.pop('type')
        assert output_type in ['mean', 'default', 'max']
        self.output_type = output_type
        # add on 
        self.global_token = TextSA(1024,0.2)
        # self.mapping_lang = torch.nn.Sequential(
        #   nn.Linear(1024, 512),
        #   nn.ReLU(),
        #   nn.Dropout(0.2),
        # )
    def forward(self, ref_expr_inds):
        """Args:
            ref_expr_inds (tensor): [batch_size, max_token], 
                integer index of each word in the vocabulary,
                padded tokens are 0s at the last.
                
        Returns:
            y (tensor): [batch_size, 1, C_l].

            y_word (tensor): [batch_size, max_token, C_l].

            y_mask (tensor): [batch_size, max_token], dtype=torch.bool, 
                True means ignored position.
        """
        # 比如前面两个是token，后面都是padding的0，整体长度为15.
        #tensor([False, False,  True,  True,  True,  True,  True,  True,  True,  True...])
        y_mask = torch.abs(ref_expr_inds) == 0 
        # refexprids : 单个句子的maxtoken是15，padding为0[TOEKNDI,TOKENID,0,0,0,0...]
        y_word = self.embedding(ref_expr_inds)
        # y_word = [batsize,maxtoken,300]
        y_word, h = self.lstm(y_word)
        # lstm投影到hiddensize维，y_word [batchsize,maxtoken=15,1024]
        #                        h     [2        ,batchsize  ,512]
        if self.output_type == "mean":
            y = torch.cat(list(map(lambda feat, mask: torch.mean(feat[mask, :], dim=0, keepdim=True), y_word, ~y_mask))).unsqueeze(1)
        elif self.output_type == "max":
            y = torch.cat(list(map(lambda feat, mask: torch.max(feat[mask, :], dim=0, keepdim=True)[0], y_word, ~y_mask))).unsqueeze(1)
        elif self.output_type == "default":
            h = h.transpose(0, 1) # [batch size ,2,512]
            y = h.flatten(1).unsqueeze(1)# flatten(x),x维及之后的数据展平
        #print(f"forward lstm y return : {y.shape}")# shape [batchsize,1,1024]
        # add on
        # template = Variable(torch.zeros(y_word.shape).cuda())

        # for i in range(y_mask.shape[0]):
        #     n_tokens = (~y_mask[i] ==True).sum() 
        #     template[i,n_tokens:,:] = 0.
        # # torch.stack(list(map(lambda feat, mask: feat[mask, :], y_word, ~y_mask)))
        # y_word = template

        final_y = self.global_token(y_word,y)
        # # return final_y.unsqueeze(1) , y_word, y_mask
        #todo::测试functional的norm有没有nan
        
        return  final_y.unsqueeze(1)
