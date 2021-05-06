# -*- coding: utf-8 -*-
# @project：test
# @author:caojinlei
# @file: low_tree.py
# @time: 2021/04/27
import copy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class Tree4Decision:
    def __init__(self):
        self.feature_name = -1
        self.feature_value = None
        self.children = {}
        self.class_label = None


class BaseDecisionTree:
    def __init__(self,plan):
        self.decision_tree = None
        self.lower_purity = 0.7
        self.plan= plan

    def show_tree(self):
        print('开始展示决策树', self.decision_tree.__dict__)
        self.show_tree_recusively(self.decision_tree)

    def show_tree_recusively(self, current_node):
        if current_node.children == {}:
            return
        else:
            for feature_value in current_node.children:
                node = current_node.children[feature_value]
                self.show_tree_recusively(node)
                print(node.__dict__)

    def cal_purity(self, output_data):
        """
        计算样本纯度，计算公式：纯度=最多类别样本量/总样本量
        :param output_data:样本标签
        :return:纯度，最多标签
        """
        label_map = {}
        for label in output_data:
            label_map[label] = label_map.get(label, 0) + 1
        label_list = sorted(label_map.items(), key=lambda x: x[1], reverse=True)
        most_label = label_list[0][0]
        purity = label_map[most_label] / len(output_data)
        return purity, most_label

    def choose_feature_with_entropy(self, input_data, feature_list):
        """
        利用信息墒选择最优信息量最大的特征
        :param input_data:输入特征值
        :param feature_list:特征名
        :return:信息量最大的特征
        ------------------------------
        example
        >>> input_data = [[0,0,0,0],[1,0,0,0],[2,2,0,0],[3,3,1,0]]
        >>> feature_list =[a,b,c,d]
        >>> choose_feature_with_entropy(input_data,feature_list)
        >>> a
        """
        number_samples = len(input_data)
        print('特征列表', feature_list)
        value_map = {}
        for line in input_data:
            for i in feature_list:
                feature_value = line[i]
                if i in value_map:
                    value_map[i][feature_value] = value_map[i].get(feature_value, 0) + 1
                else:
                    value_map[i] = {}
                    value_map[i][feature_value] = 1
        print(value_map)
        entropy_map = {}
        for feature_name in feature_list:
            value_freq = value_map[feature_name]
            feature_entropy = 0
            for feature_value in value_freq:
                num = value_freq[feature_value]
                feature_entropy -= (num / number_samples) * np.log2(num / number_samples)
                # 信息熵公式entropy= - p_1*log(p_1) - p_2*log(p_2) - ... - p_k*log(p_k)
            entropy_map[feature_name] = feature_entropy
        entory_list = sorted(entropy_map.items(), key=lambda x: x[1], reverse=True)
        best_feature = entory_list[0][0]
        return best_feature

    def chooseBestFeatureWithIGR(self, inputData, leftFeatresIndexList, outputData):
        classLabelSet = set(outputData)
        emptyClassNumMap = {}
        for classLabel in classLabelSet:
            emptyClassNumMap[classLabel] = 0.

        totalNumOfSamples = len(inputData)  # 样本的总数，用来计算某个特征取值出现的概率
        ##############开始统计各个特征的取值在样本中出现的次数#############
        # 这种统计在朴素贝叶斯等算法中是常用的，通常用来计算需要的概率
        valueSampleNumMap = {}  # 存储各个特征的各个取值下，各类样本的数量分布，用于计算按照特征取值分组后的信息熵
        classSampleNumMap = {}  # 存储各类样本的数量部分，用于计算分组之前的信息熵
        for j in range(totalNumOfSamples):  # 遍历剩下的每一个特征，计算各自对应的信息熵
            line = inputData[j]
            classLabel = outputData[j]
            classSampleNumMap[classLabel] = classSampleNumMap.get(classLabel, 0.) + 1.
            for i in leftFeatresIndexList:  # 遍历每个剩余特征的编号(就是索引值)
                featureValue = line[i]  # 当前特征的取值
                if i in valueSampleNumMap:  # 如果这个特征编号已经收录
                    if featureValue in valueSampleNumMap[i]:
                        valueSampleNumMap[i][featureValue][classLabel] += 1.
                    else:
                        valueSampleNumMap[i][featureValue] = copy.deepcopy(emptyClassNumMap)
                        valueSampleNumMap[i][featureValue][classLabel] += 1.
                else:
                    valueSampleNumMap[i] = {}
                    if i in valueSampleNumMap:  # 如果这个特征编号已经收录
                        if featureValue in valueSampleNumMap[i]:
                            valueSampleNumMap[i][featureValue][classLabel] += 1.
                        else:
                            valueSampleNumMap[i][featureValue] = copy.deepcopy(emptyClassNumMap)
                            valueSampleNumMap[i][featureValue][classLabel] += 1.
        ##############完成统计各个特征的取值在样本中出现的次数#############
        #####开始计算分组之前的信息熵######
        entropy_before_group = 0.
        for classLabel in classSampleNumMap:
            p_this_class = classSampleNumMap[classLabel] / totalNumOfSamples
            entropy_before_group -= p_this_class * np.log2(p_this_class)
        #####完成计算分组之前的信息熵######
        #######开始计算按照各个特征分组之后的信息熵###########
        entropyMap = {}
        entropyGroupByFeatreOnlyMap = {}
        for featureNO in leftFeatresIndexList:
            valueFreqMap = valueSampleNumMap[featureNO]  # 取出这个特征的各个取值水平下，各个类别的数量分布
            featureEntropy = 0.  # 这个按照这个特征分组后的信息熵
            entropyGroupByFeatreOnlyMap[featureNO] = 0.
            for featureValue in valueFreqMap:  # 计算各个取值水平对应的样本的信息熵
                numOfEachClassMap = valueFreqMap[featureValue]  # 取出这个取值水平对应样本的列别数量分布
                numOfSamplesWithFeatureValue = np.sum(list(numOfEachClassMap.values()))  # 计算这个取值水平对应的样本总数
                featureValueEntropy = 0.
                for classLabel in numOfEachClassMap:  # 遍历每一个类别
                    # 这个取值水平对应的样本中，类别为当前类别的概率
                    p_featureValue_class = numOfEachClassMap[classLabel] / numOfSamplesWithFeatureValue
                    if p_featureValue_class == 0:  # 如果这个值为0,需要跳过，避免对取数操作无意义。这个值为0表示这个组里没有样本
                        pass
                    else:
                        featureValueEntropy -= p_featureValue_class * np.log2(p_featureValue_class)
                p_feature_value = numOfSamplesWithFeatureValue / totalNumOfSamples  # 这个特征取值出现的概率
                featureEntropy += p_feature_value * featureValueEntropy
                # 信息熵公式entropy= - p_1*log(p_1) - p_2*log(p_2) - ... - p_k*log(p_k)
                entropyGroupByFeatreOnlyMap[featureNO] -= p_feature_value * np.log2(p_feature_value)
            entropyMap[featureNO] = featureEntropy
        #######完成计算按照各个特征分组之后的信息熵###########

        # 计算信息增益
        IGMap = {}
        for featureNO in entropyMap:
            IGMap[featureNO] = entropy_before_group - entropyMap[featureNO]
        # 计算信息增益比率
        IGRMap = {}
        for featureNO in entropyMap:
            IGRMap[featureNO] = entropyMap[featureNO] / entropyGroupByFeatreOnlyMap[featureNO]

        IGRList = sorted(IGRMap.items(), key=lambda x: x[1], reverse=True)  # 按照信息增益率大小倒序排列特征的编号
        IGList = sorted(IGMap.items(), key=lambda x: x[1], reverse=True)  # 按照信息增益率大小倒序排列特征的编号
        if self.plan == 'igr':
            bestFeatureNO = IGRList[0][0]  # 挑出信息增益率最大的特征编号
        else:
            bestFeatureNO = IGList[0][0]
        return bestFeatureNO

    def fit(self, input_data, output_data):
        root_node = Tree4Decision()
        feature_list = list(range(len(input_data[0])))
        self.generate_tree(input_data, output_data, root_node, feature_list)
        self.decision_tree = root_node
        self.show_tree()

    def predict(self, input_data):
        if_children = True
        children_tree = self.decision_tree
        class_label = None
        while if_children == True:
            feature_name = children_tree.feature_name
            feature_value = input_data[feature_name]
            if children_tree.children == {}:
                if_children = False
                class_label = children_tree.class_label
            else:
                children_tree = children_tree.children[feature_value]
        return class_label

    def generate_tree(self, input_data, output_data, current_node, feature_list):
        """
        生成数据
        :param input_data:
        :param output_data:
        :param current_node:
        :param feature_list:
        :return:
        """
        purity, most_label = self.cal_purity(output_data)
        if purity > self.lower_purity or feature_list == []:  # 如果纯度够高，那就不分裂了，直接将该节点的标签写为label
            current_node.class_label = most_label
            return current_node
        else:
            if self.plan == 'en':
                best_split = self.choose_feature_with_entropy(input_data, feature_list)
            else:
                best_split = self.chooseBestFeatureWithIGR(input_data,feature_list,output_data)
            current_node.feature_name = best_split
            sample_group_map = {}
            for i in range(len(output_data)):
                sample_input = input_data[i]
                sample_output = output_data[i]
                best_feature_value = sample_input[best_split]
                if best_feature_value in sample_group_map:
                    sample_group_map[best_feature_value]['input_data'].append(sample_input)
                    sample_group_map[best_feature_value]['output_data'].append(sample_output)
                else:
                    sample_group_map[best_feature_value] = {'input_data': [sample_input],
                                                            'output_data': [sample_output]}
            feature_list.remove(best_split)
            for feature_value in sample_group_map:
                thisnode = Tree4Decision()
                thisnode.feature_name, thisnode.feature_value, thisnode.children = best_split, feature_value, {}
                current_node.children[feature_value] = thisnode
                self.generate_tree(sample_group_map[feature_value]['input_data'],
                                   sample_group_map[feature_value]['output_data'],
                                   thisnode,
                                   feature_list)

    def accuracy(self, predict, real):
        """
        计算精准率
        :param predict:预测结果
        :param real:真实结果
        :return:
        """
        right_num = 0
        for i in range(len(predict)):
            if predict[i] == real[i]:
                right_num += 1
        print('分类准确率', right_num / len(predict))


def cmd_entry(plan):
    """
    主函数，不过需要对数据处理，因为这个树只能处理离散型数据
    :return:
    """
    data = load_iris()
    lines = data.data
    input_data = []
    for line in lines:
        re = list(map(lambda x: int(x), line))
        input_data.append(re)
    output_data = data.target
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)
    clf = BaseDecisionTree(plan=plan)
    clf.fit(x_train, y_train)
    preds = []
    for i in range(len(x_test)):
        pred = clf.predict(x_test[i])
        preds.append(pred)
    clf.accuracy(preds, y_test)


if __name__ == '__main__':
    cmd_entry(plan='igr')
