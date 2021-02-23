from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
import numpy as np
import fitlog
import pickle
import os
from fastNLP import cache_results
from fastNLP import Instance
# from fastNLP.embeddings import StaticEmbedding
from fastNLP_module import StaticEmbedding
import numpy as np

def get_char(ins):
    result = []
    for token in ins:
        for c in token:
            result.append(c)
    return result

def get_tar(ins):
    result = []
    for tags in ins:
        tags = tags.split('|')
        for tag in tags:
            result.append(tag)
    return result

def trans_to_int(ins):
    result = []
    for t in ins:
        result.append(int(t))
    return result


@cache_results(_cache_fp='cache/ontonotes4ner',_refresh=False)
def load_ontonotes4ner(path,char_embedding_path=None,bigram_embedding_path=None, use_dep=True, index_token=True,train_clip=False,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    if use_dep == 0:
        print("**************We use the original dataset*****************")
        train_path = os.path.join(path,'train.char.bmes{}'.format('_clip' if train_clip else ''))
        dev_path = os.path.join(path,'dev.char.bmes')
        test_path = os.path.join(path,'test.char.bmes')
        loader = ConllLoader(['chars', 'target'])
    elif use_dep == 1:
        print("***************We use the dependency dataset**************")
        train_path = os.path.join(path, 'train.dep{}'.format('_clip' if train_clip else ''))
        dev_path = os.path.join(path, 'dev.dep{}'.format('_clip' if train_clip else ''))
        test_path = os.path.join(path, 'test.dep{}'.format('_clip' if train_clip else ''))
        loader = ConllLoader(headers=['token', 'POS', 'token.head', 'token.head_label', 'targets'], indexes=[0, 1, 2, 3, 4])
    else:
        print("The value of the use_dep is chossen from 1 and 0, not other values")
        exit()

    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    if use_dep == 1:
        for k, v in datasets.items():
            v.apply_field(get_tar, field_name='targets', new_field_name='target')
            v.apply_field(trans_to_int, field_name='token.head', new_field_name='token.head')
            v.apply_field(get_char, field_name='token', new_field_name='chars')
    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    print(*list(datasets.keys()))

    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))
    for k,v in datasets.items():
         v.set_input('chars','bigrams','seq_len','target')
         v.set_target('target','seq_len')
         
    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings



@cache_results(_cache_fp='cache/resume_ner',_refresh=False)
def load_resume_ner(path,char_embedding_path=None,bigram_embedding_path=None,use_dep=True, index_token=True, train_clip=True,
                    char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    if use_dep == 0:
        print("**************We use the original dataset*****************")
        train_path = os.path.join(path,'train.char.bmes{}'.format('_clip' if train_clip else ''))
        dev_path = os.path.join(path,'dev.char.bmes')
        test_path = os.path.join(path,'test.char.bmes')
        loader = ConllLoader(['chars', 'target'])
    elif use_dep == 1:
        print("***************We use the dependency dataset**************")
        train_path = os.path.join(path, 'train.dep{}'.format('_clip' if train_clip else ''))
        dev_path = os.path.join(path, 'dev.dep{}'.format('_clip' if train_clip else ''))
        test_path = os.path.join(path, 'test.dep{}'.format('_clip' if train_clip else ''))
        loader = ConllLoader(headers=['token', 'POS', 'token.head', 'token.head_label', 'targets'], indexes=[0, 1, 2, 3, 4])
    else:
        print("The value of the use_dep is chossen from 1 and 0, not other values")
        exit()

    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    if use_dep == 1:
        for k, v in datasets.items():
            v.apply_field(get_tar, field_name='targets', new_field_name='target')
            v.apply_field(trans_to_int, field_name='token.head', new_field_name='token.head')
            v.apply_field(get_char, field_name='token', new_field_name='chars')

    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))
    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list',_refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path,drop_characters=True):
    f = open(embedding_path,'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x:len(x) != 1, w_list))

    return w_list


@cache_results(_cache_fp='cache/msraner1',_refresh=True)
def load_msra_ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,use_dep=0, train_clip=False,
                              char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0, mask_ratio=None):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams   
    if use_dep == 0:
        print("**************We use the original dataset*****************")
        train_path = os.path.join(path,'train_dev.char.bmes{}'.format('_clip' if train_clip else ''))
        test_path = os.path.join(path,'test.char.bmes{}'.format('_clip' if train_clip else ''))
        loader = ConllLoader(['chars', 'target'])
    elif use_dep == 1:
        print("***************We use the dependency dataset**************")
        train_path = os.path.join(path, 'train.dep{}'.format('_clip' if train_clip else ''))
        test_path = os.path.join(path, 'test.dep{}'.format('_clip' if train_clip else ''))
        loader = ConllLoader(headers=['token', 'POS', 'token.head', 'token.head_label', 'targets'], indexes=[0, 1, 2, 3, 4])
    else:
        print("The value of the use_dep is chossen from 1 and 0, not other values")
        exit()

    train_bundle = loader.load(train_path)
    test_bundle = loader.load(test_path)


    datasets = dict()

    from random import sample
    train_dev = train_bundle.datasets['train']
    train_dev_example = len(train_dev)
    train_example = int(train_dev_example /100 * 80)
    dev_example = train_dev_example - train_example
    l = [i for i in range(train_dev_example)]
    dev_index = sample(l, dev_example)
    from fastNLP import DataSet
    datasets['dev'] = DataSet()
    datasets['train'] = DataSet()
    for i in range(train_dev_example):
        if i in dev_index:
            datasets['dev'].append(train_dev[i])
        else:
            datasets['train'].append(train_dev[i])
    datasets['test'] = test_bundle.datasets['train']
    if use_dep == 1:
        for k, v in datasets.items():
            v.apply_field(get_tar, field_name='targets', new_field_name='target')
            v.apply_field(trans_to_int, field_name='token.head', new_field_name='token.head')
            v.apply_field(get_char, field_name='token', new_field_name='chars')

    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    print('label_vocab:{}\n{}'.format(len(label_vocab), label_vocab.idx2word))
    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings


@cache_results(_cache_fp='cache/weiboNER_uni+bi', _refresh=False)
def load_weibo_ner(path,char_embedding_path=None, bigram_embedding_path=None, use_dep=None, index_token=True, train_clip=False,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01, mask_ratio=None):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    if use_dep == 0 :
        print("**************We use the original dataset*****************")
        train_path = os.path.join(path,'weiboNER_2nd_conll.train')
        dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev')
        test_path = os.path.join(path, 'weiboNER_2nd_conll.test')
        loader = ConllLoader(['chars', 'target'])
    elif use_dep == 1:
        print("***************We use the dependency dataset**************")
        train_path = os.path.join(path, 'train.dep{}'.format('_clip' if train_clip else ''))
        dev_path = os.path.join(path, 'dev.dep{}'.format('_clip' if train_clip else ''))
        test_path = os.path.join(path, 'test.dep{}'.format('_clip' if train_clip else ''))
        loader = ConllLoader(headers=['token', 'POS', 'token.head', 'token.head_label', 'targets'], indexes=[0, 1, 2, 3, 4])
    else:
        print("The value of the use_dep is chossen from 1 and 0, not other values")
        exit()

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}
    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']

    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    if use_dep == 1:
       for k, v in datasets.items():
        v.apply_field(get_tar, field_name='targets', new_field_name='target')
        v.apply_field(trans_to_int, field_name='token.head', new_field_name='token.head')
        v.apply_field(get_char, field_name='token', new_field_name='chars')
          
    for k,v in datasets.items():
        if use_dep == 0:
            v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')
        v.add_seq_len('chars', new_field_name='seq_len')
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])

    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab



    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                   field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                  field_name='target', new_field_name='target')

    for k,v in datasets.items():
         v.set_input('chars','bigrams','seq_len','target')
         v.set_target('target','seq_len')

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, model_dir_or_name=char_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


if __name__ == '__main__':
    pass
