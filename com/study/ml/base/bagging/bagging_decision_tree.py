import pandas as pd
import numpy  as np
import pylab  as pyl

from csv import reader
from random import seed
from random import randrange

dataset = list()
with open('./sonar.all-data.csv', 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
        if not row:
            continue
        dataset.append(row)
len(dataset)
for i in range(len(dataset[0]) - 1):
    for row in dataset:
        row[i] = float(row[i].strip())

class_values = set([row[-1] for row in dataset])
lookup = dict()
for i, value in enumerate(class_values):
    lookup[value] = i
for row in dataset:
    row[-1] = lookup[row[-1]]

print('dim  dataset =', len(dataset))


# Split a dataset into k folds
def cross_validation_split(data, n_folds):
    print('Entring cross_validation_split n_folds = ', n_folds)
    dat_split = list()
    dim = len(data)
    print('dim =', dim)
    dat_copy = list(data)
    fold_size = int(dim / n_folds)
    print('fold_size =', fold_size)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dat_copy))
            # print('index = ', index)
            fold.append(dat_copy.pop(index))
        dat_split.append(fold)
    print('Exiting cross_validation_split n_folds = ')
    return dat_split


def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, data):
    left, right = list(), list()
    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, class_values):
    gini = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += (proportion * (1.0 - proportion))
    return gini


# Select the best split point for a dataset
def get_split(data):
    class_values = list(set(row[-1] for row in data))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(data[0]) - 1):
        for row in data:
            groups = test_split(index, row[index], data)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    # print('b_index =', b_index)
    # print('b_value =', b_value)
    # print('b_score =', b_score)
    # print('dim groups =', len(groups))
    # print('dim right=', len(b_groups[0]))
    # print('dim left=', len(b_groups[1]))
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    # print('1 length root = ',len(root))
    split(root, max_depth, min_size, 1)
    # print_tree(root)
    return root


# Make prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Bootstrap aggregation algorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(data, algorithm, n_folds, *args):
    folds = cross_validation_split(data, n_folds)
    print('evaluation_algorithm: number folds = ', len(folds))
    scores = list()

    for f in range(len(folds)):
        print('***********fold =', f + 1, '****************')
        train_set = list(folds)
        fold = folds[f]
        del train_set[f]
        # Todo sum([[]],[])
        # 由于上一步已经 del train_set[f] 删除测试集,然后sum(train_set, [])拼接所有train_set
        # 构成一个新的测试接
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    print(' evaluation_algorithm: number folds = ', len(folds))
    return scores


seed(1)
n_folds = 5
max_depth = 6
min_size = 2
sample_size = 0.50
for n_trees in [1, 5, 10, 50]:
    scores = evaluate_algorithm(dataset, bagging, n_folds, max_depth, min_size, sample_size, n_trees)
    print(n_trees, scores)
print('Trees: %d' % n_trees)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))