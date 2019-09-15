import sys
import os
import jieba
import pandas as pd

'''
label->id
'''

train_file = "/Users/piguanghua/Downloads/cnews/cnews.train.txt"
val_file = "/Users/piguanghua/Downloads/cnews/cnews.val.txt"
test_file = "/Users/piguanghua/Downloads/cnews/cnews.test.txt"

#output_file
seg_train_file = "/Users/piguanghua/Downloads/cnews/cnews.train.seg.txt"
seg_val_file = "/Users/piguanghua/Downloads/cnews/cnews.train.val.txt"
seg_test_file = "/Users/piguanghua/Downloads/cnews/cnews.train.test.txt"
vocab_file = "/Users/piguanghua/Downloads/cnews/cnews.train.vocab.txt"
category_file = "/Users/piguanghua/Downloads/cnews/category.txt"


def generate_seg_file(input_file, output_seg_file):
    with open(input_file, 'r') as fr:
        lines = fr.readlines()
    with open(output_seg_file, 'w') as fw:
        for line in lines:
            label, content = line.strip("\r\n").split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(" ")
                if word != ' ':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            fw.write(out_line)

'''
with open(val_file, 'r') as f:
    lines = f.readlines()
    label,content = lines[0].strip("\r\n").split('\t')
    word_iter = jieba.cut(content)
    print( '/'.join(word_iter) )
'''

def generate_vacab_file(input_seg_file, output_vacab_file):
    with open(input_seg_file, 'r') as f:
        lines = f.readlines()
        word_dict = {}
        for line  in lines:
            label, content = line.strip('\r\n').split('\t')
            for word in content.split():
                word_dict.setdefault(word, 0)
                word_dict[word] += 1
        sorted_word_dict = sorted(
            word_dict.items(), key = lambda d:d[1], reverse = True
        )
        with open(output_vacab_file, 'w') as f:
            f.write("<UNK>\t10000000\n")
            for item in sorted_word_dict:
                f.write("%s\t%d\n" %(item[0], item[1]))


def generate_category_dict(input_file, category_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    category_dict = {}
    for line in lines:
        label, content = line.split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] +=1
    #category_number = len(category_dict)
    with open(category_file, 'w') as f:
        for category in category_dict:
            line = '%s\n' % category
            print("%s\t%d" % (category, category_dict[category]))
            f.write(line)




if __name__ == '__main__':
    generate_seg_file(train_file, seg_train_file)
    generate_seg_file(val_file, seg_val_file)
    generate_seg_file(test_file, seg_test_file)

    generate_vacab_file(seg_train_file, vocab_file)

    generate_category_dict(train_file, category_file)
