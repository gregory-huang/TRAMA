import os
import torch.nn as nn
import torch
from V_withbert.modules import Transformer_Encoder
from fastNLP.modules import ConditionalRandomField
import collections
from utils import get_crf_zero_init
from fastNLP import seq_len_to_mask
from utils import print_info
from utils import better_init_rnn
from fastNLP.modules import LSTM
import math
import copy
from utils import size2MB
from utils import MyDropout

def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


class Absolute_SE_Position_Embedding(nn.Module):
    def __init__(self,fusion_func,hidden_size,learnable,mode=collections.defaultdict(bool),pos_norm=False,max_len=5000,):
        '''

        :param fusion_func:暂时只有add和concat(直接拼接然后接线性变换)，
        后续得考虑直接拼接再接非线性变换，和将S和E两个位置做非线性变换再加或拼接
        :param hidden_size:
        :param learnable:
        :param debug:
        :param pos_norm:
        :param max_len:
        '''
        super().__init__()
        self.fusion_func = fusion_func
        assert self.fusion_func in {'add','concat','nonlinear_concat','nonlinear_add','add_nonlinear','concat_nonlinear'}
        self.pos_norm = pos_norm
        self.mode = mode
        self.hidden_size = hidden_size
        pe = get_embedding(max_len,hidden_size)
        pe_sum = pe.sum(dim=-1,keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        # pe = pe.unsqueeze(0)
        pe_s = copy.deepcopy(pe)
        pe_e = copy.deepcopy(pe)
        self.pe_s = nn.Parameter(pe_s, requires_grad=learnable)
        self.pe_e = nn.Parameter(pe_e, requires_grad=learnable)
        if self.fusion_func == 'concat':
            self.proj = nn.Linear(self.hidden_size * 3,self.hidden_size)

        if self.fusion_func == 'nonlinear_concat':
            self.pos_proj = nn.Sequential(nn.Linear(self.hidden_size * 2,self.hidden_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_size,self.hidden_size))
            self.proj = nn.Linear(self.hidden_size * 2,self.hidden_size)

        if self.fusion_func == 'nonlinear_add':
            self.pos_proj = nn.Sequential(nn.Linear(self.hidden_size * 2,self.hidden_size),
                                          nn.LeakyReLU(),
                                          nn.Linear(self.hidden_size,self.hidden_size))

        if self.fusion_func == 'concat_nonlinear':
            self.proj = nn.Sequential(nn.Linear(self.hidden_size * 3,self.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size))

        if self.fusion_func == 'add_nonlinear':
            self.proj = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.hidden_size,self.hidden_size))


    def forward(self,inp,pos_s,pos_e):
        batch = inp.size(0)
        max_len = inp.size(1)
        pe_s = self.pe_s.index_select(0, pos_s.view(-1)).view(batch,max_len,-1)
        pe_e = self.pe_e.index_select(0, pos_e.view(-1)).view(batch,max_len,-1)

        if self.fusion_func == 'concat':
            inp = torch.cat([inp,pe_s,pe_e],dim=-1)
            output = self.proj(inp)
        elif self.fusion_func == 'add':
            output = pe_s + pe_e + inp
        elif self.fusion_func == 'nonlinear_concat':
            pos = self.pos_proj(torch.cat([pe_s,pe_e],dim=-1))
            output = self.proj(torch.cat([inp,pos],dim=-1))
        elif self.fusion_func == 'nonlinear_add':
            pos = self.pos_proj(torch.cat([pe_s,pe_e],dim=-1))
            output = pos + inp
        elif self.fusion_func == 'add_nonlinear':
            inp = inp + pe_s + pe_e
            output = self.proj(inp)

        elif self.fusion_func == 'concat_nonlinear':
            output = self.proj(torch.cat([inp,pe_s,pe_e],dim=-1))



        return output

        # if self.fusion_func == 'add':
        #     result =

    def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        rel pos init:
        如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
        如果是1，那么就按-max_len,max_len来初始化
        """
        num_embeddings = 2 * max_seq_len + 1
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        if rel_pos_init == 0:
            emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        else:
            emb = torch.arange(-max_seq_len, max_seq_len + 1, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

class Absolute_Position_Embedding(nn.Module):
    def __init__(self,fusion_func,hidden_size,learnable,mode=collections.defaultdict(bool),pos_norm=False,max_len=5000):
        '''

        :param hidden_size:
        :param max_len:
        :param learnable:
        :param debug:
        :param fusion_func:暂时只有add和concat(直接拼接然后接线性变换)，后续得考虑直接拼接再接非线性变换
        '''
        super().__init__()
        self.fusion_func = fusion_func
        assert ('add' in self.fusion_func) != ('concat' in self.fusion_func)
        if 'add' in self.fusion_func:
            self.fusion_func = 'add'
        else:
            self.fusion_func = 'concat'
        #备注，在SE绝对位置里，会需要nonlinear操作来融合两种位置pos，但普通的不需要，所以只根据字符串里有哪个关键字来判断
        self.pos_norm = pos_norm
        self.mode = mode
        self.debug = mode['debug']
        self.hidden_size = hidden_size
        pe = get_embedding(max_len,hidden_size)
        # pe = torch.zeros(max_len, hidden_size,requires_grad=True)
        # position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0, hidden_size, 2,dtype=torch.float32) *
        #                      -(math.log(10000.0) / float(hidden_size)))
        # pe[:, 0::2] = torch.sin((position * div_term))
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe_sum = pe.sum(dim=-1,keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=learnable)

        if self.fusion_func == 'concat':
            self.proj = nn.Linear(self.hidden_size * 2,self.hidden_size)

        if self.mode['debug']:
            print_info('position embedding:')
            print_info(self.pe[:100])
            print_info('pe size:{}'.format(self.pe.size()))
            print_info('pe avg:{}'.format(torch.sum(self.pe)/(self.pe.size(2)*self.pe.size(1))))
    def forward(self,inp):
        batch = inp.size(0)
        if self.mode['debug']:
            print_info('now in Absolute Position Embedding')
        if self.fusion_func == 'add':
            output = inp + self.pe[:,:inp.size(1)]
        elif self.fusion_func == 'concat':
            inp = torch.cat([inp,self.pe[:,:inp.size(1)].repeat([batch]+[1]*(inp.dim()-1))],dim=-1)
            output = self.proj(inp)

        return output

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb


class Sandwich_Transformer_SeqLabel(nn.Module):
    def __init__(self,lattice_embed, bigram_embed, hidden_size, label_size,
                 num_heads, num_layers, masks,
                 use_abs_pos,use_rel_pos, learnable_position,add_position,
                 layer_preprocess_sequence, layer_postprocess_sequence,
                 ff_size=-1, scaled=True , dropout=None,use_bigram=True,mode=collections.defaultdict(bool),
                 dvc=None,vocabs=None, 
                 require_ffn=None,
                 use_dep_control=0, max_seq_len=-1,
                 rel_pos_shared=True,k_proj=True,q_proj=True,v_proj=True,r_proj=True,
                 attn_ff=True,pos_norm=False,ff_activate='relu',rel_pos_init=0,
                 abs_pos_fusion_func='concat',embed_dropout_pos='0',
                 four_pos_shared=True,four_pos_fusion=None,four_pos_fusion_shared=True,bert_embedding=None):
        super().__init__()
        self.masks = masks
        self.use_bert = False
        if bert_embedding is not None:
            self.use_bert = True
            self.bert_embedding = bert_embedding

        self.four_pos_fusion_shared = four_pos_fusion_shared
        self.mode = mode
        self.use_dep_control = use_dep_control
        self.four_pos_shared = four_pos_shared
        self.abs_pos_fusion_func = abs_pos_fusion_func
        self.lattice_embed = lattice_embed
        self.bigram_embed = bigram_embed
        self.hidden_size = hidden_size
        self.label_size = label_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert four_pos_fusion is not None
        self.four_pos_fusion = four_pos_fusion
        self.learnable_position = learnable_position
        self.add_position = add_position
        self.rel_pos_shared = rel_pos_shared
        self.vocabs = vocabs
        self.attn_ff = attn_ff
        self.pos_norm = pos_norm
        self.ff_activate = ff_activate
        self.rel_pos_init = rel_pos_init
        self.embed_dropout_pos = embed_dropout_pos


        if self.use_rel_pos and max_seq_len < 0:
            print_info('max_seq_len should be set if relative position encode')
            exit(1208)

        self.max_seq_len = max_seq_len

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.pe = None

        if self.use_abs_pos:
            self.abs_pos_encode = Absolute_SE_Position_Embedding(self.abs_pos_fusion_func,
                                        self.hidden_size,learnable=self.learnable_position,mode=self.mode,
                                        pos_norm=self.pos_norm)

        if self.use_rel_pos:
            pe = get_embedding(
                max_seq_len,hidden_size,rel_pos_init=self.rel_pos_init)
            pe_sum = pe.sum(dim=-1,keepdim=True)
            if self.pos_norm:
                with torch.no_grad():
                    pe = pe/pe_sum
            self.pe = nn.Parameter(pe, requires_grad=self.learnable_position)
            if self.four_pos_shared:
                self.pe_ss = self.pe
                self.pe_se = self.pe
                self.pe_es = self.pe
                self.pe_ee = self.pe
            else:
                self.pe_ss = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_se = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_es = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
                self.pe_ee = nn.Parameter(copy.deepcopy(pe),requires_grad=self.learnable_position)
        else:
            self.pe = None
            self.pe_ss = None
            self.pe_se = None
            self.pe_es = None
            self.pe_ee = None


        self.layer_preprocess_sequence = layer_preprocess_sequence
        self.layer_postprocess_sequence = layer_postprocess_sequence
        if ff_size==-1:
            ff_size = self.hidden_size
        self.ff_size = ff_size
        self.scaled = scaled
        if dvc == None:
            dvc = 'cpu'
        self.dvc = torch.device(dvc)
        if dropout is None:
            self.dropout = collections.defaultdict(int)
        else:
            self.dropout = dropout
        self.use_bigram = use_bigram

        if self.use_bigram:
            self.bigram_size = self.bigram_embed.embedding.weight.size(1)
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)+self.bigram_embed.embedding.weight.size(1)
        else:
            self.char_input_size = self.lattice_embed.embedding.weight.size(1)

        if self.use_bert:
            self.char_input_size+=self.bert_embedding._embed_size

        self.lex_input_size = self.lattice_embed.embedding.weight.size(1)

        self.embed_dropout = MyDropout(self.dropout['embed'])
        self.gaz_dropout = MyDropout(self.dropout['gaz'])

        self.char_proj = nn.Linear(self.char_input_size,self.hidden_size)
        self.lex_proj = nn.Linear(self.lex_input_size,self.hidden_size)

        self.encoder = Transformer_Encoder(self.hidden_size,self.num_heads,
                                           self.num_layers,
                                           relative_position=self.use_rel_pos,
                                           learnable_position=self.learnable_position,
                                           add_position=self.add_position,
                                           layer_preprocess_sequence=self.layer_preprocess_sequence,
                                           layer_postprocess_sequence=self.layer_postprocess_sequence,
                                           dropout=self.dropout,
                                           scaled=self.scaled,
                                           ff_size=self.ff_size,
                                           masks=self.masks, 
                                           require_ffn=require_ffn,
                                           use_dep_control=self.use_dep_control,
                                           mode=self.mode,
                                           dvc=self.dvc,
                                           max_seq_len=self.max_seq_len,
                                           pe=self.pe,
                                           pe_ss=self.pe_ss,
                                           pe_se=self.pe_se,
                                           pe_es=self.pe_es,
                                           pe_ee=self.pe_ee,
                                           k_proj=self.k_proj,
                                           q_proj=self.q_proj,
                                           v_proj=self.v_proj,
                                           r_proj=self.r_proj,
                                           attn_ff=self.attn_ff,
                                           ff_activate=self.ff_activate,
                                           lattice=True,
                                           four_pos_fusion=self.four_pos_fusion,
                                           four_pos_fusion_shared=self.four_pos_fusion_shared)

        self.output_dropout = MyDropout(self.dropout['output'])
        self.output = nn.Linear(self.hidden_size,self.label_size)
        self.crf = get_crf_zero_init(self.label_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        self.target_label = 0
        self.output_feature = 0

    def forward(self, lattice, bigrams, seq_len, lex_num, pos_s, pos_e,
                target, dep_matrix=None):

        batch_size = lattice.size(0)
        total_length = lattice.size(1)
        max_seq_len = bigrams.size(1)

        raw_embed = self.lattice_embed(lattice)
        if self.use_bigram:
            bigrams_embed = self.bigram_embed(bigrams)
            bigrams_embed = torch.cat([bigrams_embed,
                                       torch.zeros(size=[batch_size,total_length-max_seq_len,
                                                         self.bigram_size]).to(bigrams_embed)],dim=1)
            raw_embed_char = torch.cat([raw_embed, bigrams_embed],dim=-1)
        else:
            raw_embed_char = raw_embed

        if self.use_bert:
            bert_pad_length = lattice.size(1)-max_seq_len
            char_for_bert = lattice[:, :max_seq_len]
            mask = seq_len_to_mask(seq_len).bool()
            char_for_bert = char_for_bert.masked_fill((~mask),self.vocabs['lattice'].padding_idx)
            bert_embed = self.bert_embedding(char_for_bert)
            bert_embed = torch.cat([bert_embed,
                                    torch.zeros(size=[batch_size,bert_pad_length,bert_embed.size(-1)],
                                                device = bert_embed.device,
                                                requires_grad=False)],dim=-2)
            raw_embed_char = torch.cat([raw_embed_char, bert_embed],dim=-1)


        if self.embed_dropout_pos == '0':
            raw_embed_char = self.embed_dropout(raw_embed_char)
            raw_embed = self.gaz_dropout(raw_embed)

        '''
        seq_len_to_mask [True,...,True,False,...False]
        masked_fill:try to replace the value whose position is True
        '''
        embed_char = self.char_proj(raw_embed_char)
        char_mask = seq_len_to_mask(seq_len,max_len=total_length).bool()
        embed_char.masked_fill_(~(char_mask.unsqueeze(-1)), 0)

        embed_lex = self.lex_proj(raw_embed)
        lex_mask = (seq_len_to_mask(seq_len+lex_num).bool() ^ char_mask.bool())
        embed_lex.masked_fill_(~(lex_mask).unsqueeze(-1), 0)

        assert char_mask.size(1) == lex_mask.size(1)

        embedding = embed_char + embed_lex
        if self.embed_dropout_pos == '1':
            embedding = self.embed_dropout(embedding)

        if self.use_abs_pos:
            embedding = self.abs_pos_encode(embedding,pos_s,pos_e)

        if self.embed_dropout_pos == '2':
            embedding = self.embed_dropout(embedding)
        encoded = self.encoder(
            embedding,seq_len, total_length,
            lex_num=lex_num,pos_s=pos_s,
            pos_e=pos_e, dep_matrix=dep_matrix,
        )

        if hasattr(self,'output_dropout'):
            encoded = self.output_dropout(encoded)
        encoded = encoded[:,:max_seq_len,:]
        pred = self.output(encoded)

        mask = seq_len_to_mask(seq_len).bool()
        if self.training:
            self.output_feature = encoded
            self.target_label = target
            loss = self.crf(pred, target, mask).mean(dim=0)
            return {'loss': loss}
        else:
            pred, path = self.crf.viterbi_decode(pred, mask)
            result = {'pred': pred}
            return result

