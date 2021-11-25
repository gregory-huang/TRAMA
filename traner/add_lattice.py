from fastNLP import cache_results
import numpy as np
import copy

@cache_results(_cache_fp='need_to_defined_fp',_refresh=True)
def equip_chinese_ner_with_lexicon(datasets,vocabs,embeddings,w_list,word_embedding_path=None,
                                   use_dep=None, 
                                   data_name=None,
                                   only_lexicon_in_train=False,word_char_mix_embedding_path=None,
                                   number_normalized=False,
                                   lattice_min_freq=1,only_train_min_freq=0):
    '''
    The dataset transfered here has many fields like
    ['token', 'POS', 'token.head', 'token.head_label', 'targets', 'chars', 'bigrams', 'seq_len']
    '''
    from fastNLP.core import Vocabulary
    def normalize_char(inp):
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result

    def normalize_bigram(inp):
        result = []
        for bi in inp:
            tmp = bi
            if tmp[0].isdigit():
                tmp = '0'+tmp[:1]
            if tmp[1].isdigit():
                tmp = tmp[0]+'0'

            result.append(tmp)
        return result

    if number_normalized == 3:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result
    from traner.utils_ import Trie
    from functools import partial
    from fastNLP.core import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP import DataSet
    a = DataSet()
    # a.apply
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)


    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s,e,lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)


    def get_dep_matrix(ins):
        result = []
        token_head = ins['token.head']
        token = ins['token']
        lattice = ins['lattice']
        pos_s = ins['pos_s']
        pos_e = ins['pos_e']
        seq_len = ins['seq_len']
        total_seq_len = len(ins['lattice'])
        for i, l in enumerate(lattice):
            if i>0:
                result.append([i, i - 1])
            if i<len(lattice):
                result.append([i, i + 1])

        ''''
        pos_s = ins['pos_s']
        pos_e = ins['pos_e']
        for i in range(seq_len):
            if i > 0:
                result.append([i, i-1])
            if i < seq_len-1:
                result.append([i, i+1])
        for i in range(seq_len, total_seq_len):
            for j in range(i+1, total_seq_len):
                if pos_s[j] <= pos_e[i]:
                    result.append([i,j])
                    result.append([j,i])
        '''

        if data_name == 'weibo':
            return result
        # Now we add the dependency relations
        token_pos = []
        s = 0
        e = 0
        for t in token:
            e += len(t)
            token_pos.append([s,e-1])
            s += len(t)
        for i in range(len(token)):
            if 0 < token_head[i] < len(token):
                t_s, t_e = token_pos[i]
                t_h_s, t_h_e = token_pos[token_head[i]-1]
                a = b = -1
                for j in range(len(lattice)):
                    if t_s == pos_s[j] and t_e == pos_e[j]:
                        a = j
                    if t_h_s == pos_s[j] and t_h_e == pos_e[j]:
                        b = j
                    if a != -1 and b != -1:
                        result.append([a, b])
                        result.append([b, a])
                        break
        return result

    import copy
    for k,v in datasets.items():
        v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','lexicons')
        v.apply_field(copy.copy, 'chars','raw_chars')
        v.apply_field(copy.copy, 'target', 'raw_target')
        v.add_seq_len('lexicons','lex_num')
        v.apply_field(lambda x:list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')


    if number_normalized == 1:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if number_normalized == 2:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x:x[2],lexicons))
        # print('lexicons:{}'.format(lexicons))
        # print('lex_only:{}'.format(list(filter(lambda x:x[2],lexicons))))
        # print('result:{}'.format(result))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e

        return pos_e

    def get_chars_pos(ins):
        import numpy as np
        token = ins['token']
        chars = ins['chars']
        pos = ins['POS']
        char_pos = []
        for i, t in enumerate(token):
            if len(t) == 1:
                char_pos.append(pos[i])
            else:
                for _ in range(len(t)):
                    char_pos.append('P'+pos[i])
        assert len(char_pos) == ins['seq_len']
        return char_pos

    for k,v in datasets.items():
        v.apply(concat,new_field_name='lattice')
        v.apply(get_pos_s,new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('lattice')
        v.set_input('pos_s','pos_e')
        if use_dep == 1:
            v.apply(get_dep_matrix, new_field_name='dep_matrix')
            v.set_input('dep_matrix')
    if use_dep == 1:
        pass
        '''for k,v in datasets.items():        
                v.delete_field('token')
                v.delete_field('token.head')
                v.delete_field('token.head_label')'''


    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab
    #print(datasets['train'][0].fields)
    #lexicons_vocab = Vocabulary()
    #lexicons_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='lexicons')
    #vocabs['lexicons'] = lexicons_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'],field_name='lattice',
                               no_create_entry_dataset=[v for k,v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab
    
    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path,word_dropout=0.01,
                                            min_freq=lattice_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    vocabs['char'].index_dataset(* (datasets.values()),
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(* (datasets.values()),
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(* (datasets.values()),
                              field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(* (datasets.values()),
                                    field_name='lattice', new_field_name='lattice')

    return datasets,vocabs,embeddings
