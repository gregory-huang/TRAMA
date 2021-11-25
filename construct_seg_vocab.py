from fastNLP.io.loader import ConllLoader
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='weibo', help='weibo|resume|ontonotes|msra')
args = parser.parse_args()
word_dataset_path = {
    'weibo':'/home/hzy/ChineseNER/word_datasets/weibo',
    'ontonotes':'/home/hzy/ChineseNER/word_datasets/ontonotes',
    'resume':'/home/hzy/ChineseNER/word_datasets/resume',
    'msra':'/home/hzy/ChineseNER/word_datasets/msra'
}
datasets = dict()

loader = ConllLoader(['words', 'target'])
dataset_path = word_dataset_path[args.dataset]
train_path = os.path.join(dataset_path, 'train.word.bmes')
train_bundle = loader.load(train_path)
datasets['train'] = train_bundle.datasets['train']
if args.dataset != 'msra':
    dev_path = os.path.join(dataset_path, 'dev.word.bmes')
    dev_bundle = loader.load(dev_path)
    datasets['dev'] = dev_bundle.datasets['train']

test_path = os.path.join(dataset_path, 'test.word.bmes')
test_bundle = loader.load(test_path)
datasets['test'] = test_bundle.datasets['train']



seg_word_list = list()
for k,v in datasets.items():
    for ins in v:
        word_sequence = ins['words']
        for w in word_sequence:
            if len(w) > 1 and w not in seg_word_list:
                seg_word_list.append(w)

seg_word_list = list(set(seg_word_list))
seg_vocab_path = os.path.join(dataset_path, 'seg_vocab.txt')

with open(seg_vocab_path, 'w') as f:
    for word in seg_word_list:
        content = word + '\n'
        f.writelines(content)
f.close()


