import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import math
# this is the test for github
df_train = pd.read_csv('data/cleaned_training_data.csv').iloc[0:100, :]
df_test = pd.read_csv('data/cleaned_testing_data.csv')

# np.random.seed(12345)

# random sample dataset
def sampleData(dataset, frac):
    return dataset.sample(frac=frac, replace=True)


# find groups for the given index and value
def findGroup(dataset, index, value):
    left_group = []
    right_group = []
    for i in range(len(dataset)):
        if dataset.iloc[i, index] < value:
            left_group.append(dataset.iloc[i])
        else:
            right_group.append(dataset.iloc[i])
    return [left_group, right_group]


# calculate the gini indexªª
def giniIndex(dataset, classes):
    # print('ginigroup', dataset)
    sum_value = dataset.groupby(classes).count().iloc[:, 0]
    #     print('sum_value', sum_value)
    accu = []
    size = len(dataset)
    for value in sum_value:
        accu.append((value / size) ** 2)
    #     print('accu',accu)
    gini = 1 - sum(accu)
    #     print('gini',gini_D)
    return gini


# calculate the impurate gini index
def impurateGiniIndex(dataset, groups, classes):
    left_df = groups[0]
    right_df = groups[1]
    gini_D = giniIndex(dataset, classes)
    if len(left_df) == 0:
        left_gini = 1
    else:
        left_df = pd.concat(left_df, axis=1).transpose()
        left_gini = giniIndex(left_df, classes)
    if len(right_df) == 0:
        right_gini = 1
    else:
        right_df = pd.concat(right_df, axis=1).transpose()
        right_gini = giniIndex(right_df, classes)

    gini_A = len(left_df) / len(dataset) * left_gini + len(right_df) / len(dataset) * right_gini
    delta_gini = gini_D - gini_A

    return delta_gini


# find best split point
def bestSplitPoint(dataset, max_features):
    if isinstance(dataset, list):
        dataset = pd.concat(dataset, axis=1).transpose()
    indexs_n = len(list(dataset)) - 1
    r_indexs = np.random.choice(indexs_n, size=max_features, replace=False)
    # print('r_indexs', r_indexs)
    classes = dataset.iloc[:, -1]
    best_index = 0
    best_groups = None
    best_value = 0
    largest_gini = -1
    for index in r_indexs:
        values = dataset.iloc[:, index].unique()
        for value in values:
            # print('index', index, 'value', value)
            groups = findGroup(dataset, index, value)
            #             print('groups',groups)
            #             print(len(groups))
            gini_index = impurateGiniIndex(dataset, groups, classes)
            # print('gini',gini_index)
            if gini_index > largest_gini:
                largest_gini = gini_index
                best_value = value
                best_groups = groups
                best_index = index
    # print('node', {'index': best_index, 'value': best_value, 'groups': best_groups})
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


# find the class of the group
def terminate(group):
    classes = pd.concat(group, axis=1).transpose().iloc[:, -1]
    return classes.mode()[0]


def split(node, max_depth, min_sample_leaf, max_features, depth):
    #     print('groups', node['groups'])
    # print('groups', len(node['groups']))
    left = node['groups'][0]
    right = node['groups'][1]
    del node['groups']

    if (left == []) | (right == []):
        # print('!!!!!!!!!',left + right)
        node['left'] = terminate(left + right)
        node['right'] = terminate(left + right)
        return None

    if depth >= max_depth:
        node['left'] = terminate(left)
        node['right'] = terminate(right)
        return None

    if len(left) <= min_sample_leaf:
        node['left'] = terminate(left)
    else:
        node['left'] = bestSplitPoint(left, max_features)
        split(node['left'], max_depth, min_sample_leaf, max_features, depth + 1)

    if len(right) <= min_sample_leaf:
        node['right'] = terminate(right)
    else:
        node['right'] = bestSplitPoint(right, max_features)
        split(node['right'], max_depth, min_sample_leaf, max_features, depth + 1)


# construct the tree
def decisionTree(data, max_depth, min_sample_leaf, max_features):
    root = bestSplitPoint(data, max_features)
    split(root, max_depth, min_sample_leaf, max_features, 1)
    return root


# make prediction
def predict(node, entry):
    if entry[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], entry)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], entry)
        else:
            return node['right']


# create random forest
def randomForest(train_data, test_data, max_depth, min_sample_leaf, max_features, min_sample_split, n_estimators):
    frac = 1 / min_sample_split
    trees = []
    t_predictions = []
    predictions = []
    for i in range(n_estimators):
        # print('#####################', i)
        sample_data = sampleData(train_data, frac)
        sample_data = sample_data.reset_index(drop = True)
        tree = decisionTree(sample_data, max_depth, min_sample_leaf, max_features)
        trees.append(tree)
    for i in range(len(test_data)):
        for tree in trees:
            t_predictions.append(predict(tree, test_data.iloc[i]))
        predictions.append(round(sum(t_predictions) / len(t_predictions)))
    print(predictions)
    return predictions




# test
xtrain, xtest, ytrain, ytest = train_test_split(df_train.iloc[:,0: -1], df_train.iloc[:, -1], test_size=0.33, random_state=42)
xtrain['results'] = ytrain
pred = randomForest(xtrain, xtest, max_depth = 8, min_sample_leaf = 33, max_features = round(math.sqrt(len(list(xtrain)) - 1)), min_sample_split = 5, n_estimators = 450)

# accuracy
print ('accuracy: ', accuracy_score(ytest, pred))
