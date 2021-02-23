from paths import *
from V_withbert.load_data import *
from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--clip_length',
    default={
        ontonote4ner_cn_path:150,
        resume_ner_path:200,
        weibo_ner_path: 160,
        msra_ner_cn_path: 200
    }
)
parser.add_argument('--mask_thereshold', default=0.25)
parser.add_argument('--generate_ex_nums', default=5)
args = parser.parse_args()
import numpy as np

def dep_generation_ner(path,train_clip=False):
    from fastNLP.io.loader import ConllLoader
    from fastNLP.core import Vocabulary
    from fastHan import FastHan
    model = FastHan(model_type='base')

    def get_dep(sentence):
        sentences = ''.join(char for char in sentence)
        result = []
        answer = model(sentences, 'Parsing')
        for i, sentence in enumerate(answer):
            for token in sentence:
                result.append([token, token.pos, token.head, token.head_label])
        return result
    if path == msra_ner_cn_path:
        train_path = os.path.join(path,'train_dev.char.bmes{}'.format('_clip' if train_clip else ''))
    else:
        train_path = os.path.join(path,'train.char.bmes{}'.format('_clip' if train_clip else ''))
    if path == msra_ner_cn_path:
        dev_path = os.path.join(path,'test.char.bmes')
    else:
        dev_path = os.path.join(path,'dev.char.bmes')
    test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']



    for k,v in datasets.items():
        fp_out = os.path.join(path,'{}.dep'.format(k))
        f_out = open(fp_out, 'w', encoding='utf-8')
        sentences = v.chars.content
        tags = v.target.content
        for i, sentence in enumerate(sentences):
            tag = tags[i]
            try:
                result = get_dep(sentence)
            except Exception as e:
                print(sentence)
                result = []
                for n in range(len(sentence)):
                    r = []
                    r.append(sentence[n])
                    r.append('o-POS')
                    r.append(n)
                    r.append('no-dep')
            pos_s = 0
            pos_e = 0
            for line in result:
                word = [c for c in str(line[0])]
                pos_e += len(word)
                tag_for_word = tag[pos_s:pos_e]
                pos_s = pos_e
                line1 = '\t'.join(str(c) for c in line)
                line2 = '|'.join(str(c) for c in tag_for_word)
                line = line1 + '\t' + line2
                print(line, end='\n', file=f_out)
            line = '\n'
            print(line, end='', file=f_out)

def dep_generation_ner4weibo(path,train_clip=False):
    from fastNLP.io.loader import ConllLoader
    from fastNLP.core import Vocabulary
    from fastHan import FastHan
    model = FastHan(model_type='large')

    def get_dep(sentence):
        sentences = ''.join(char for char in sentence)
        result = []
        answer = model(sentences, 'Parsing')
        for i, sentence in enumerate(answer):
            for token in sentence:
                result.append([token, token.pos, token.head, token.head_label])
        return result

    def normalize(chars):
        from string import digits
        result = []
        for c in chars:
            c=c.translate(str.maketrans('', '', digits))
            result.append(c)
        return result

    def _to_utf8(filePathSrc):
        import os
        import sys
        from Npp import notepad # import it first!

        for root, dirs, files in os.walk(filePathSrc):
            for fn in files: 
                if fn[-4:] == '.xml': # Specify type of the files
                    notepad.open(root + "\\" + fn)      
                    notepad.runMenuCommand("Encoding", "Convert to UTF-8")
                    # notepad.save()
                    # if you try to save/replace the file, an annoying confirmation window would popup.
                    notepad.saveAs("{}{}".format(fn[:-4], '_utf8.xml')) 
                    notepad.close()
    train_path = os.path.join(path, 'weiboNER_2nd_conll.train')
    dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev')
    test_path = os.path.join(path, 'weiboNER_2nd_conll.test')

    #train_path = os.path.join(path, 'train.char.bmes')
    #dev_path = os.path.join(path, 'dev.char.bmes')
    #test_path = os.path.join(path, 'test.char.bmes')

    #train_path = os.path.join(path, 'weiboNER.conll.train')
    #dev_path = os.path.join(path, 'weiboNER.conll.dev')
    #test_path = os.path.join(path, 'weiboNER.conll.test')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    for k, v in datasets.items():
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')


    for k,v in datasets.items():
        fp_out = os.path.join(path,'{}.dep'.format(k))
        f_out = open(fp_out, 'w', encoding='utf-8')
        sentences = v.chars.content
        tags = v.target.content
        for i, sentence in enumerate(sentences):
            tag = tags[i]
            try:
                result = get_dep(sentence)
            except Exception as e:
                print(sentence)
                result = []
                for n in range(len(sentence)):
                    r = []
                    r.append(sentence[n])
                    r.append('o-POS')
                    r.append(n)
                    r.append('no-dep')
            pos_s = 0
            pos_e = 0
            for line in result:
                word = [c for c in str(line[0])]
                pos_e += len(word)
                tag_for_word = tag[pos_s:pos_e]
                pos_s = pos_e
                line1 = '\t'.join(str(c) for c in line)
                line2 = '|'.join(str(c) for c in tag_for_word)
                line = line1 + '\t' + line2
                print(line, end='\n', file=f_out)
            line = '\n'
            print(line, end='', file=f_out)

def create_cliped_dep_file(path):
    from fastNLP.io.loader import ConllLoader
    from fastNLP.core import Vocabulary

    train_dep_path = os.path.join(path, 'train.dep')
    dev_dep_path = os.path.join(path, 'dev.dep')
    test_dep_path = os.path.join(path, 'test.dep')

    loader = ConllLoader(headers=['token', 'POS', 'token_head', 'token_head_label', 'targets'], indexes=[0, 1, 2, 3, 4])
    train_dep_bundle = loader.load(train_dep_path)
    dev_dep_bundle = loader.load(dev_dep_path)
    test_dep_bundle = loader.load(test_dep_path)

    datasets_dep = dict()
    datasets_dep['train'] = train_dep_bundle.datasets['train']
    datasets_dep['dev'] = dev_dep_bundle.datasets['train']
    datasets_dep['test'] = test_dep_bundle.datasets['train']

    for k, v in datasets_dep.items():
        fp_out = os.path.join(path, '{}.dep_clip'.format(k))
        f_out = open(fp_out, 'w', encoding='utf-8')
        #v.apply_field(get_dep_char, field_name='token', new_field_name='chars')
        token = v.token.content
        POS = v.POS.content
        token_head = v.token_head.content
        token_head_label = v.token_head_label.content
        targets = v.targets.content
        for i in range(len(token)):
            t = token[i]
            p = POS[i]
            th = token_head[i]
            thl = token_head_label[i]
            target = targets[i]
            l_t = 0
            l_char = 0
            sparse_time = []
            while(l_t < len(t)):
                if l_char >= args.clip_length[path]:
                    line = '\n'
                    print(line, end='', file=f_out)
                    l_char = 0
                    sparse_time.append(l_t)
                if len(sparse_time) > 0:
                    line_content = [t[l_t], p[l_t], int(th[l_t])-sparse_time[-1], thl[l_t], target[l_t]]
                else:
                    line_content = [t[l_t], p[l_t], th[l_t], thl[l_t], target[l_t]]
                line = '\t'.join(str(c) for c in line_content)
                print(line, end='\n', file=f_out)
                l_char += len(t[l_t])
                l_t += 1
            line = '\n'
            print(line, end='', file=f_out)

def create_cliped_dep_masked_file(path, clip=True):
    from fastNLP.io.loader import ConllLoader
    from fastNLP.core import Vocabulary

    train_dep_path = os.path.join(path, 'train.dep{}'.format('_clip' if clip else ''))
    dev_dep_path = os.path.join(path, 'dev.dep{}'.format('_clip' if clip else ''))
    test_dep_path = os.path.join(path, 'test.dep{}'.format('_clip' if clip else ''))

    loader = ConllLoader(headers=['token', 'POS', 'token_head', 'token_head_label', 'targets'], indexes=[0, 1, 2, 3, 4])
    train_dep_bundle = loader.load(train_dep_path)
    dev_dep_bundle = loader.load(dev_dep_path)
    test_dep_bundle = loader.load(test_dep_path)

    datasets_dep = dict()
    datasets_dep['train'] = train_dep_bundle.datasets['train']
    datasets_dep['dev'] = dev_dep_bundle.datasets['train']
    datasets_dep['test'] = test_dep_bundle.datasets['train']

    for k, v in datasets_dep.items():
        # 遍历每一个dataset
        fp_out = os.path.join(path, '{}.dep_clip_masked_{}'.format(k, args.mask_thereshold))
        f_out = open(fp_out, 'w', encoding='utf-8')
        v.apply_field(get_dep_char, field_name='token', new_field_name='chars')
        token = v.token.content
        POS = v.POS.content
        token_head = v.token_head.content
        token_head_label = v.token_head_label.content
        targets = v.targets.content
        '''
        为了增加训练样例的多样性，
        考虑在模型训练时不给模型提供完全的entity_name，
        而是将entity对应的词性（或者直接为mask）提供给模型，
        希望模型可以学得context_level information
        '''
        # 遍历每一个dataset中每一个instance
        for i in range(len(token)):
            # write the original examples to the file
            t = token[i]
            p = POS[i]
            th = token_head[i]
            thl = token_head_label[i]
            target = targets[i]
            l_t = 0
            entity_nums = 0
            while (l_t < len(t)):
                target_for_token = target[l_t].split('|')
                for tt in target_for_token:
                    if tt != 'O':
                        entity_nums += 1
                        break
                t_tem = str(t[l_t]).replace('|', '-')
                t_tem = '|'.join(x for x in t_tem)
                line_content = [t_tem, p[l_t], th[l_t], thl[l_t], target[l_t]]
                line = '\t'.join(str(c) for c in line_content)
                print(line, end='\n', file=f_out)
                l_t += 1
            line = '\n'
            print(line, end='', file=f_out)
            # write the degenerate examples
            # entity_masked = [0 for _ in range(len(t))]

            for _ in range(min(args.generate_ex_nums, entity_nums)):
                l_t = 0
                is_entity = False
                while (l_t < len(t)):
                    target_for_token = target[l_t].split('|')
                    for tt in target_for_token:
                        if tt != 'O':
                            is_entity = True
                    thereshold = np.random.uniform(0, 1)
                    if not is_entity and thereshold <= args.mask_thereshold:
                        if len(t[l_t]) == 1:
                            replacement_token = p[l_t]
                        else:
                            replacement_token = '|'.join(x for x in ['P' + p[l_t] for _ in range(len(t[l_t]))])

                    elif is_entity and thereshold <= ((np.sqrt(entity_nums*len(t))) / (len(t)+ entity_nums)) :
                        if len(t[l_t]) == 1:
                            replacement_token = p[l_t]
                        else:
                            replacement_token = '|'.join(x for x in ['P' + p[l_t] for _ in range(len(t[l_t]))])
                    else:
                        replacement_token = '|'.join(x for x in t[l_t])
                    line_content = [replacement_token, p[l_t], th[l_t], thl[l_t], target[l_t]]
                    line = '\t'.join(str(c) for c in line_content)
                    print(line, end='\n', file=f_out)
                    l_t += 1
                line = '\n'
                print(line, end='', file=f_out)

#dep_generation_ner(ontonote4ner_cn_path)
#dep_generation_ner(resume_ner_path)
#dep_generation_ner4weibo(weibo_ner_path)
#dep_generation_ner(msra_ner_cn_path)

#create_cliped_dep_file(ontonote4ner_cn_path)
create_cliped_dep_file(weibo_ner_path)
#create_cliped_dep_file(resume_ner_path)
#create_cliped_dep_file(msra_ner_cn_path)

#create_cliped_dep_masked_file(ontonote4ner_cn_path)
#create_cliped_dep_masked_file(weibo_ner_path)
