import os
import sys 
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from seqtr.models import LAN_ENCODERS

@LAN_ENCODERS.register_module()
class BertEncoder(nn.Module):
    def __init__(self,word_emb,num_token, num_layers = 2,bert_name="bert-base-uncased"):
        super(BertEncoder, self).__init__()
        self.num_layers = num_layers
        self.bert_name = bert_name
        
        if self.bert_name == 'bert-base-uncased':
            self.lang_dim = 768
        else:
            self.lang_dim = 1024
        
        self.bert = BertModel.from_pretrained(self.bert_name)#,output_hidden_states=True

        for parameter in self.bert.encoder.layer[:11].parameters():
            parameter.requires_grad_(False)
        self.mapping = nn.Sequential(nn.Linear(self.lang_dim, 1024))
                # nn.BatchNorm1d(1024), 
                # nn.ReLU(inplace=True), 
                # nn.Dropout(0.2),
                # nn.Linear(1024, 1024),
                # nn.BatchNorm1d(1024),
                # nn.ReLU(inplace=True))
    def forward(self, ref_expr_inds, attention_mask=None):
        y = self.bert.embeddings.word_embeddings(ref_expr_inds)
        y = self.mapping(y)
        
        # import pdb;pdb.set_trace()
        # output_ = self.bert(ref_expr_inds, attention_mask)
        # # if self.num_layers > 0:
        # #     y = output_[self.num_layers - 1]
        # #     import pdb;pdb.set_trace()
        # # else:
        # #     y = self.mapping(output_.pooler_output)
        return y , ~attention_mask.to(torch.bool)
