import numpy as np
import pandas as pd


from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from math import log


'''
    def createBranch(数据集，decision_tree)
        判断数据集中是否只有Feature
            if 数据集唯一 return
            else
                计算数据集ShanCnt
                找BestFeature
                切分数据集
                for 数据集
                    createBranch(数据集)
                return decision_tree
        
'''

def create_dataset():
	dataSet = [[0, 0, 0, 0, 'no'],						#数据集
			[0, 0, 0, 1, 'no'],
			[0, 1, 0, 1, 'yes'],
			[0, 1, 1, 0, 'yes'],
			[0, 0, 0, 0, 'no'],
			[1, 0, 0, 0, 'no'],
			[1, 0, 0, 1, 'no'],
			[1, 1, 1, 1, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[1, 0, 1, 2, 'yes'],
			[2, 0, 1, 2, 'yes'],
			[2, 0, 1, 1, 'yes'],
			[2, 1, 0, 1, 'yes'],
			[2, 1, 0, 2, 'yes'],
			[2, 0, 0, 0, 'no']]
	labels = ['年龄', '有工作', '有自己的房子', '信贷情况', 'target']		#特征标签
	return dataSet, labels

def calc_shannon_ent(data, target="target"):
    category_probability = []
    for category in np.unique(data[target]):
        category_sum = (data[target][data[target] == category]).shape[0]
        category_probability.append(category_sum / data.shape[0])

    return (-1) * np.sum(np.array(category_probability) * np.array(np.log(np.array(category_probability))))

def calc_best_feature(data, target="target"):
    original_shan = calc_shannon_ent(data)
    total_size = data.shape[0]
    attribute_dict = {}
    attributes = data.columns.tolist()
    attributes.remove(target)
    for attribute in attributes:
        attribute_shan = 0
        for attribute_item in np.unique(data[attribute]):
            size_x = (data[attribute][data[attribute] == attribute_item]).shape[0]
            for target_item in np.unique(data[target]):
                x_y_num = data[ (data[target] == target_item) & (data[attribute] ==  attribute_item) ].shape[0]
                if x_y_num == 0:
                    pass
                else:
                    attribute_shan += x_y_num / total_size * log( (x_y_num / total_size)/(size_x /total_size ) )

        attribute_dict[attribute] = (-1) * attribute_shan

    max_shan = 0e10
    best_attribute = None
    for key,value in attribute_dict.items():
        information_gain = original_shan - value
        if max_shan < information_gain:
            best_attribute = key
            max_shan = information_gain

    return best_attribute

def first_n(df,n=3):
    n = df.shape[0]
    return df[0:n]

def split_data(data, feature):
    group_data = data.groupby(feature).apply(first_n)
    group_data.drop(columns=[feature])
    split_data_list = {}
    for level in group_data.index.levels[0].tolist():
        split_data_list[group_data.index.names[0]+str(level)] = group_data.loc[level]
    return split_data_list


def create_decision_tree(data,target="target"):
    if len(data[target][data[target] == "yes"]) == len(data):
        return "yes"
    elif len(data[target][data[target] == "no"]) == len(data):
        return "no"
    else:
        #base_shan = calc_shannon_ent(data)
        best_feature = calc_best_feature(data)
        split_data_list = split_data(data, best_feature)
        #decision_tree={best_feature:{}}
        myTree = {best_feature: {}}

        for (key,value) in split_data_list.items():
            myTree[best_feature][key] = create_decision_tree(value)

        return myTree


if __name__ == '__main__':
    dataSet,labels = create_dataset()
    df = pd.DataFrame(dataSet)
    df.columns = labels
    decision_tree = {}
    print(create_decision_tree(df))



