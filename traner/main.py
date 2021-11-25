import fitlog
use_fitlog = True
if not use_fitlog:
    fitlog.debug()

load_dataset_seed = 100
#fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')
#fitlog.set_rng_seed(load_dataset_seed)
import sys
sys.path.append('../')
from traner.load_data import *
import argparse
import numpy as np
from traner.paths import *
from traner.trainer import Trainer
from fastNLP.io import ModelSaver, ModelLoader
from fastNLP import Tester
from fastNLP.core import Callback, RandomSampler
from traner.models import Sandwich_Transformer_SeqLabel
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric, MetricBase
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback
from fastNLP import FitlogCallback
from fastNLP import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import fitlog
from fastNLP import logger
from utils import get_peking_time
from traner.add_lattice import equip_chinese_ner_with_lexicon
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import traceback
import warnings
import sys
from utils import print_info
import copy
#from fastNLP_module import BertEmbedding
from fastNLP.embeddings import BertEmbedding
parser = argparse.ArgumentParser()
# performance inrelevant
parser.add_argument('--update_every', type=int, default=1)
parser.add_argument('--status', choices=['train', 'test'], default='train')
parser.add_argument('--fix_bert_epoch', type=int,default=0)
parser.add_argument('--train_clip', default=True)
parser.add_argument('--device', default='1')
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=1080956, type=int)
parser.add_argument('--number_normalized', type=int, default=0, choices=[0,1,2,3])
parser.add_argument('--lexicon_name', default='yj')
parser.add_argument('--use_pytorch_dropout', type=int, default=0)
parser.add_argument('--gpumm', default=False, help='查看显存')
parser.add_argument('--char_min_freq', default=1, type=int)
parser.add_argument('--bigram_min_freq', default=1, type=int)
parser.add_argument('--lattice_min_freq', default=1, type=int)
parser.add_argument('--only_train_min_freq', default=True)
parser.add_argument('--only_lexicon_in_train', default=False)
parser.add_argument('--word_min_freq', default=1, type=int)

parser.add_argument('--early_stop', default=50, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=10, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--bert_lr_rate', default=0.05, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='norm', help='norm|uniform')
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--norm_embed', default=True)
parser.add_argument('--norm_lattice_embed', default=True)
parser.add_argument('--warmup', default=0, type=float)


parser.add_argument('--model', default='transformer', help='lstm|transformer')
parser.add_argument('--lattice', default=1, type=int)
parser.add_argument('--use_bigram', default=1, type=int)
parser.add_argument('--hidden', default=-1, type=int)
parser.add_argument('--ff', default=3, type=int)
parser.add_argument('--head', default=8,type=int)
parser.add_argument('--head_dim', default=16, type=int)
parser.add_argument('--scaled', default=False)
parser.add_argument('--ff_activate', default='relu', help='leaky|relu')
parser.add_argument('--k_proj', default=True)
parser.add_argument('--q_proj', default=True)
parser.add_argument('--v_proj', default=True)
parser.add_argument('--r_proj', default=True)

parser.add_argument('--attn_ff', default=False)

parser.add_argument('--use_abs_pos', default=False)
parser.add_argument('--use_rel_pos', default=True)

parser.add_argument('--rel_pos_shared', default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm', default=False)
parser.add_argument('--rel_pos_init', default=1)
parser.add_argument('--four_pos_shared', default=True)
parser.add_argument('--four_pos_fusion', default='ff_two', choices=['ff', 'attn', 'gate', 'ff_two', 'ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared',default=True,help='是不是要共享4个位置融合之后形成的pos')
parser.add_argument('--post', default='an')


parser.add_argument('--embed_dropout_before_pos',default=False)
parser.add_argument('--embed_dropout', default=0.5,type=float)
parser.add_argument('--gaz_dropout',default=0.5,type=float)
parser.add_argument('--output_dropout', default=0.5,type=float)
parser.add_argument('--pre_dropout', default=0.5,type=float)
parser.add_argument('--post_dropout', default=0.5,type=float)
parser.add_argument('--ff_dropout', default=0.15,type=float)
parser.add_argument('--ff_dropout_2', default=-1,type=float,help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout',default=0,type=float)
parser.add_argument('--embed_dropout_pos',default='0')
parser.add_argument('--abs_pos_fusion_func',default='nonlinear_add',
                    choices=['add','concat','nonlinear_concat','nonlinear_add','concat_nonlinear','add_nonlinear'])


# hyper of model
parser.add_argument('--use_bert', type=int, default=0)
parser.add_argument('--use_dep_control', default='s')
parser.add_argument('--require_ffn', default=[], type=list)
parser.add_argument('--masks', default=['no_use'], type=list)
parser.add_argument('--layer', default=2, type=int)
parser.add_argument('--use_dep', default=1, type=int)
parser.add_argument('--pre', default='')
parser.add_argument('--dataset', default='weibo', help='weibo|resume|ontonotes|msra')
parser.add_argument('--record_after', default=0, type=int)


args = parser.parse_args()
if args.use_dep == 1:
    args.masks = ['dep_and_adj'] * (args.layer - 1) + ['no_use']
else:
    args.masks = ['no_use'] * args.layer
args.layer = len(args.masks)
args.require_ffn = [0] * (args.layer - 1)
# for _ in range(args.layer-1):
args.require_ffn.append(1)
require_ffn = args.require_ffn
if args.dataset == 'weibo':
    args.update_every = 2
    args.train_clip = True
    args.batch = 12
    args.optim = 'sgd'
    args.lr = 1e-3
    args.head = 6
    args.head_dim = 16
    args.epoch = 200
    args.early_stop = 40
    args.warmup = 0/args.epoch
    
elif args.dataset == 'resume':
    args.update_every = 2
    args.train_clip = True
    args.fix_bert_epoch = 0
    args.batch = 12
    args.optim = 'sgd'
    args.lr = 1e-3
    args.head = 6
    args.head_dim = 16
    args.epoch = 150
    args.warmup = 10 / args.epoch
    args.use_dep_control = 'd'

elif args.dataset == 'ontonotes':
    args.update_every = 2
    args.train_clip = True
    args.batch = 16
    args.optim = 'sgd'
    args.lr = 1e-3
    args.head = 8
    args.head_dim = 20
    args.ff = 1
    args.epoch = 100
    args.warmup = 10/args.epoch
    # args.record_after = 58
    args.use_dep_control = 'd'

elif args.dataset == 'msra':  
    args.update_every = 2
    args.train_clip = True
    args.batch = 10
    args.optim = 'sgd'
    args.lr = 1e-3
    args.head = 8
    args.head_dim = 20
    args.ff = 1
    args.epoch = 100
    args.warmup = 10/args.epoch
    args.early_stop = 50
    args.use_dep_control = 'd'


if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout
fitlog.set_log_dir('logs/{}/'.format(args.dataset))
now_time = get_peking_time()
logger.add_file('log/{}_{}'.format(args.dataset, now_time),level='info')
if args.test_batch == -1:
    args.test_batch = args.batch//2
fitlog.add_hyper(now_time, 'time')

if args.device!= 'cpu':
    assert args.device.isdigit()
    assert torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = True
for k,v in args.__dict__.items():
    print_info('{}:{}'.format(k,v))

raw_dataset_cache_name = os.path.join('cache',args.dataset+
                          '_trainClip:{}'.format(args.train_clip)
                                      +'usedep_{}'.format(args.use_dep)
                                      +'bgminfreq_{}'.format(args.bigram_min_freq)
                                      +'char_min_freq_{}'.format(args.char_min_freq)
                                      +'word_min_freq_{}'.format(args.word_min_freq)
                                      +'only_train_min_freq{}'.format(args.only_train_min_freq)
                                      +'number_norm{}'.format(args.number_normalized)
                                      + 'load_dataset_seed{}'.format(load_dataset_seed))


if args.dataset == 'ontonotes':
    datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    use_dep=args.use_dep,
                                                    _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
                                                    _cache_fp=raw_dataset_cache_name,
                                                    char_min_freq=args.char_min_freq,
                                                    bigram_min_freq=args.bigram_min_freq,
                                                    only_train_min_freq=args.only_train_min_freq,
                                                    )
elif args.dataset == 'resume':
    datasets,vocabs,embeddings = load_resume_ner(resume_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                 _cache_fp=raw_dataset_cache_name,
                                                 char_min_freq=args.char_min_freq,
                                                 bigram_min_freq=args.bigram_min_freq,
                                                 only_train_min_freq=args.only_train_min_freq,
                                                    )
elif args.dataset == 'weibo':
    datasets,vocabs,embeddings = load_weibo_ner(weibo_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                use_dep=args.use_dep,
                                                train_clip=args.train_clip,
                                                    _refresh=refresh_data,index_token=False,
                                                _cache_fp=raw_dataset_cache_name,
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq,
                                                    )
elif args.dataset == 'msra':
    datasets,vocabs,embeddings = load_msra_ner(msra_ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                use_dep=args.use_dep,
                                                train_clip=args.train_clip,
                                               _refresh=refresh_data,index_token=False,
                                               _cache_fp=raw_dataset_cache_name,
                                               char_min_freq=args.char_min_freq,
                                               bigram_min_freq=args.bigram_min_freq,
                                              only_train_min_freq=args.only_train_min_freq,
                                                           )

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout
args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff
if args.lexicon_name == 'lk':
    yangjie_rich_pretrain_word_path = lk_word_path_2

print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name))

cache_name = os.path.join('cache',(args.dataset+'_lattice'+'_only_train:{}'+
                          '_trainClip:{}'+'_norm_num:{}'
                                   +'char_min_freq{}'+'bigram_min_freq{}'+'word_min_freq{}'+'only_train_min_freq{}'
                                   +'number_norm{}'+'lexicon_{}'+'load_dataset_seed_{}')
                          .format(args.only_lexicon_in_train,
                          args.train_clip,args.number_normalized,args.char_min_freq,
                                  args.bigram_min_freq,args.word_min_freq,args.only_train_min_freq,
                                  args.number_normalized,args.lexicon_name,load_dataset_seed))
datasets,vocabs,embeddings = equip_chinese_ner_with_lexicon(datasets,vocabs,embeddings,
                                                            w_list,yangjie_rich_pretrain_word_path,
                                                            use_dep=args.use_dep,
                                                            data_name=args.dataset,
                                                         _refresh=refresh_data,_cache_fp=cache_name,
                                                         only_lexicon_in_train=args.only_lexicon_in_train,
                                                            word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            number_normalized=args.number_normalized,
                                                            lattice_min_freq=args.lattice_min_freq,
                                                            only_train_min_freq=args.only_train_min_freq)

see_lexicon_stat = False
if see_lexicon_stat:
    print("There are {} latent words in lexicons used".format(len(w_list)))
    seg_vocab_path = {
        "weibo": '/home/hzy/ChineseNER/traner/datasets/Weibo/seg_vocab.txt',
        "msra": '/home/hzy/ChineseNER/traner/datasets/MSRA/seg_vocab.txt',
        "ontonotes": '/home/hzy/ChineseNER/traner/datasets/OntoNotes4NER/seg_vocab.txt',
        "resume": '/home/hzy/ChineseNER/traner/datasets/ResumeNER/seg_vocab.txt',
    }
    vocab_path = seg_vocab_path[args.dataset]
    print(vocab_path)
    seg_word_list = load_dataset_seg_vocab(vocab_path) # filter the character
    print(seg_word_list[0:5])
    # todo 统计有多少在seg_vocab中的词语不在word list中，所以没有cover到
    print("There are {} words in {} dataset".format(len(seg_word_list), args.dataset))
    un_coverd = 0
    for w in seg_word_list:
        if w not in w_list:
            un_coverd += 1
    print("********** There are {} words exists in {} dataset but not in lexicon **********".format(un_coverd, args.dataset))

    # todo 统计有多少在word list中的词语被匹配到了，但是没有出现在seg_vcoab中，即相当于错误的匹配了
    matched_word = list(vocabs['lattice']._word2idx.keys())
    matched_word = list(filter(lambda x: len(x) != 1, matched_word))
    mismatched = 0
    for m_w in matched_word:
        if m_w not in seg_word_list:
            mismatched += 1
    print("******** We have matched {} different words from lexicons and {} of them are wrongly matched ********".format(len(matched_word)-2, mismatched))
    exit()



max_seq_len = max(* map(lambda x:max(x['seq_len']),datasets.values()))
for k, v in datasets.items():
    if args.lattice:
        v.set_input('lattice','bigrams','seq_len','target')
        v.set_input('lex_num','pos_s','pos_e')
        v.set_target('target','seq_len', 'raw_chars', 'lattice', 'lex_num')
        v.set_pad_val('lattice',vocabs['lattice'].padding_idx)


from utils import norm_static_embedding
# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed>0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)

if args.norm_lattice_embed > 0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)


mode = {}
mode['debug'] = args.debug
mode['gpumm'] = args.gpumm
if args.debug or args.gpumm:
    fitlog.debug()
dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

#torch.backends.cudnn.benchmark = False
#fitlog.set_rng_seed(args.seed)
#torch.backends.cudnn.benchmark = False


if args.use_bert == 1:
    bert_embedding = BertEmbedding(
        vocabs['lattice'],
        model_dir_or_name='cn-wwm',
        requires_grad=True,
        word_dropout=0.01
    )
else:
    bert_embedding = None
model = Sandwich_Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden,
                                      len(vocabs['label']),
                                      args.head, args.layer, args.masks, args.use_abs_pos, args.use_rel_pos,
                                      args.learn_pos, args.add_pos,
                                      args.pre, args.post, args.ff, args.scaled, dropout, args.use_bigram,
                                      mode, device, vocabs,
                                      require_ffn=require_ffn,
                                      use_dep_control=args.use_dep_control,
                                      max_seq_len=max_seq_len,
                                      rel_pos_shared=args.rel_pos_shared,
                                      k_proj=args.k_proj, q_proj=args.q_proj,
                                      v_proj=args.v_proj, r_proj=args.r_proj,
                                      attn_ff=args.attn_ff, pos_norm=args.pos_norm,
                                      ff_activate=args.ff_activate,abs_pos_fusion_func=args.abs_pos_fusion_func,
                                      embed_dropout_pos=args.embed_dropout_pos,
                                      four_pos_shared=args.four_pos_shared,
                                      four_pos_fusion=args.four_pos_fusion,
                                      four_pos_fusion_shared=args.four_pos_fusion_shared,
                                      bert_embedding=bert_embedding,
                                      )
for n,p in model.named_parameters():
    print('{}:{}'.format(n,p.size()))
fitlog.add_hyper(args)
with torch.no_grad():
    print_info('{}init pram{}'.format('*'*15,'*'*15))
    for n,p in model.named_parameters():
        if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    print_info('xavier uniform init:{}'.format(n))
                elif args.init == 'norm':
                    print_info('xavier norm init:{}'.format(n))
                    nn.init.xavier_normal_(p)
            except:
                print_info(n)
                exit(1208)
    print_info('{}init pram{}'.format('*' * 15, '*' * 15))

loss = LossInForward()
encoding_type = 'bmeso'
if args.dataset == 'weibo':
    encoding_type = 'bio'
f1_metric = SpanFPreRecMetric(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type=encoding_type)

from datetime import datetime
start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
save_eval_res ="./eval_res/{}/{}/".format(args.dataset, start_time)

from fastNLP.core.utils import seq_len_to_mask

class SaveAccuracyMetric(MetricBase):
    def __init__(self, save_eval_res=None, pred=None, target=None, seq_len=None, label_vocab=None, lattice_vocab=None):
        super().__init__()
        self._init_param_map(pred=pred, target=target, seq_len=seq_len)
        self.total = 0
        self.acc_count = 0
        self.save_eval_res = save_eval_res
        if not os.path.exists(self.save_eval_res):
            os.mkdir(self.save_eval_res)
        self.file_count = 0
        self.label_vocab = label_vocab
        self.lattice_vocab = lattice_vocab

    def evaluate(self, pred, target, seq_len=None, raw_chars=None, lattice=None, lex_num=None):
        if self.save_eval_res !=None:
            EVAL_DIR = self.save_eval_res + "pred_res_on_epoch_" + str(self.file_count) + '.txt'
            f = open(EVAL_DIR, 'a')
            batch_size, max_len = pred.size()
            seq_len_to_mask(seq_len=seq_len, max_len=max_len)
            for i in range(batch_size):
                length = seq_len[i].cpu().numpy()
                length_lex = lex_num[i].cpu().numpy()
                chars = raw_chars[i]
                pre = pred[i].tolist()
                tar = target[i].tolist()
                lat = lattice[i].tolist()
                for j in range(length):
                    pre_token = self.label_vocab.to_word(pre[j])
                    tar_token = self.label_vocab.to_word(tar[j])
                    content = chars[j] + '\t' + pre_token + '\t' + tar_token + '\n'
                    f.write(content)
                lex_str = ''
                for l in lat[length: length + length_lex + 1]:
                    lex_str += self.lattice_vocab.to_word(l) + '\t'
                lex_str += str(length_lex) + '\n'
                f.write(lex_str)
            f.close()
        if seq_len is not None and target.dim() > 1:
            max_len = target.size(1)
            masks = seq_len_to_mask(seq_len=seq_len, max_len=max_len)
        else:
            masks = None

        target = target.to(pred)
        if masks is not None:
            self.acc_count += torch.sum(torch.eq(pred, target).masked_fill(masks.eq(False), 0)).item()
            self.total += torch.sum(masks).item()
        else:
            self.acc_count += torch.sum(torch.eq(pred, target)).item()
            self.total += np.prod(list(pred.size()))

    def get_metric(self, reset=True):
        evaluate_result = {'acc': round(float(self.acc_count) / (self.total + 1e-12), 6)}
        if reset:
            self.acc_count = 0
            self.total = 0
            self.file_count += 1
        return evaluate_result


acc_metric = SaveAccuracyMetric(
    save_eval_res=save_eval_res,
    pred='pred',
    target='target',
    seq_len='seq_len',
    label_vocab=vocabs['label'],
    lattice_vocab=vocabs['lattice']
)
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]
print("Metric established")
#datasets['train'].apply
print(torch.rand(size=[3,3],device=device))
sampler = RandomSampler()
if args.status == 'train':
    cross_validate = False
    if cross_validate is True:
        # TODO 5 cross-fold validation
        from fastNLP import DataSet
        all_dataset = DataSet()
        for v in datasets.values():
            for ins in v:
                all_dataset.append(ins)
        # TODO redistributed the datasets
        n_fold = 5
        dataset_folds = []
        for _ in range(n_fold):
            dataset_folds.append(DataSet())
        for ins in all_dataset:
            r = int(np.random.rand() * n_fold)
            dataset_folds[r].append(ins)
        print("The datasets has been distributed in {} folds".format(n_fold))
        for index, d in enumerate(dataset_folds):
            print("fold {} has {} instances".format(index, len(d)))
        datasets_train = [DataSet() for _ in range(n_fold)]
        datasets_test = []
        for test_fold in range(n_fold):
            datasets_test.append(dataset_folds[test_fold])
            for train_fold in range(n_fold):
                if train_fold != test_fold:
                    for ins in dataset_folds[train_fold]:
                        datasets_train[test_fold].append(ins)
        # TODO Copy The empty Model and the optimizer
        models = [copy.deepcopy(model) for _ in range(n_fold)]
        optimizers = []
        for i in range(n_fold):
            if args.use_bert != 1:
                print("We are not using bert embedding")
                bigram_embedding_param = list(models[i].bigram_embed.parameters())
                gaz_embedding_param = list(models[i].lattice_embed.parameters())
                embedding_param = bigram_embedding_param
                if args.lattice:
                    gaz_embedding_param = list(models[i].lattice_embed.parameters())
                    embedding_param = embedding_param + gaz_embedding_param
                embedding_param_ids = list(map(id, embedding_param))
                non_embedding_param = list(filter(lambda x: id(x) not in embedding_param_ids, models[i].parameters()))
                param_ = [{'params': non_embedding_param},
                          {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]
            else:
                print("We will use bert embedding")
                bert_embedding_param = list(models[i].bert_embedding.parameters())
                bert_embedding_param_ids = list(map(id, bert_embedding_param))
                bigram_embedding_param = list(models[i].bigram_embed.parameters())
                gaz_embedding_param = list(models[i].lattice_embed.parameters())
                embedding_param = bigram_embedding_param
                if args.lattice:
                    gaz_embedding_param = list(models[i].lattice_embed.parameters())
                    embedding_param = embedding_param + gaz_embedding_param
                embedding_param_ids = list(map(id, embedding_param))
                non_embedding_param = list(filter(
                    lambda x: id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
                    models[i].parameters()))
                param_ = [{'params': non_embedding_param},
                          {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
                          {'params': bert_embedding_param, 'lr': args.bert_lr_rate * args.lr}]
            if args.optim == 'adam':
                optimizer = optim.AdamW(param_, lr=args.lr, weight_decay=args.weight_decay)
            elif args.optim == 'sgd':
                optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            optimizers.append(optimizer)

        # TODO Train five times
        for i in range(n_fold):
            datasets_train[i].set_input('lattice', 'bigrams', 'seq_len', 'target','lex_num', 'pos_s', 'pos_e', 'dep_matrix')
            datasets_train[i].set_target('target', 'seq_len', 'raw_chars', 'lattice', 'lex_num')
            datasets_train[i].set_pad_val('lattice', vocabs['lattice'].padding_idx)
            datasets_test[i].set_input('lattice', 'bigrams', 'seq_len', 'target', 'lex_num', 'pos_s', 'pos_e', 'dep_matrix')
            datasets_test[i].set_target('target', 'seq_len', 'raw_chars', 'lattice', 'lex_num')
            datasets_test[i].set_pad_val('lattice', vocabs['lattice'].padding_idx)

            lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizers[i], lambda ep: 1 / (1 + 0.05 * ep)))
            clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
            callbacks = [lrschedule_callback, clip_callback]
            callbacks.append(EarlyStopCallback(args.early_stop))
            callbacks.append(WarmupCallback(warmup=args.warmup))
            trainer = Trainer(datasets_train[i], models[i], optimizers[i], loss, args.batch,
                              n_epochs=args.epoch,
                              dev_data=datasets_test[i],
                              metrics=metrics,
                              device=device,
                              sampler=sampler,
                              callbacks=callbacks, dev_batch_size=args.test_batch,
                              test_use_tqdm=False, check_code_level=-1, record_after=args.record_after,
                              update_every=args.update_every,
                              save_path="./save/{}_model".format(args.dataset),
                              )
            trainer.train()
    else:
        if args.use_bert != 1:
            print("We are not using bert embedding")
            bigram_embedding_param = list(model.bigram_embed.parameters())
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = bigram_embedding_param
            if args.lattice:
                gaz_embedding_param = list(model.lattice_embed.parameters())
                embedding_param = embedding_param+gaz_embedding_param
            embedding_param_ids = list(map(id,embedding_param))
            non_embedding_param = list(filter(lambda x:id(x) not in embedding_param_ids,model.parameters()))
            param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]
        else:
            print("We will use bert embedding")
            bert_embedding_param = list(model.bert_embedding.parameters())
            bert_embedding_param_ids = list(map(id,bert_embedding_param))
            bigram_embedding_param = list(model.bigram_embed.parameters())
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = bigram_embedding_param
            if args.lattice:
                gaz_embedding_param = list(model.lattice_embed.parameters())
                embedding_param = embedding_param+gaz_embedding_param
            embedding_param_ids = list(map(id, embedding_param))
            non_embedding_param = list(filter(
                lambda x: id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
                                              model.parameters()))
            param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
                      {'params':bert_embedding_param, 'lr':args.bert_lr_rate*args.lr}]

        if args.optim == 'adam':
            optimizer = optim.AdamW(param_,lr=args.lr,weight_decay=args.weight_decay)
        elif args.optim == 'sgd':
            optimizer = optim.SGD(param_,lr=args.lr,momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        print("********** optimizer created **********")
        fitlog_evaluate_dataset = {'test':datasets['dev']}
        evaluate_callback = FitlogCallback(fitlog_evaluate_dataset, verbose=1)
        lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep)))
        clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
        class Unfreeze_Callback(Callback):
            def __init__(self,bert_embedding,fix_epoch_num):
                super().__init__()
                self.bert_embedding = bert_embedding
                self.fix_epoch_num = fix_epoch_num
                assert self.bert_embedding.requires_grad == False

            def on_epoch_begin(self):
                if self.epoch == self.fix_epoch_num+1:
                    self.bert_embedding.requires_grad = True

        callbacks = [
                evaluate_callback,
                lrschedule_callback,
                clip_callback,
            ]
        if args.use_bert:
            if args.fix_bert_epoch != 0:
                callbacks.append(Unfreeze_Callback(bert_embedding,args.fix_bert_epoch))
            else:
                bert_embedding.requires_grad = True
        callbacks.append(EarlyStopCallback(args.early_stop))
        callbacks.append(WarmupCallback(warmup=args.warmup))
        print("********** callbacks created **********")
        trainer = Trainer(
            datasets['train'], model,optimizer,loss,args.batch,
            n_epochs=args.epoch,
            dev_data=datasets['test'],
            metrics=metrics,
            device=device,
            sampler=sampler,
            callbacks=callbacks,dev_batch_size=args.test_batch,
            test_use_tqdm=False,check_code_level=-1, record_after=args.record_after,
            update_every=args.update_every,
            save_path="./save/{}_model".format(args.dataset),
        )
        trainer.train()
        tester = Tester(datasets['test'], model, metrics=metrics)
        tester.test()
