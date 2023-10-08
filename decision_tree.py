import numpy as np


class Node:

    def __init__(self, feature: str, left: 'Node', right: 'Node', parent: 'Node', positives: int, negatives: int,
                 threshold: float):
        self.feature = feature
        self.left = left
        self.right = right
        self.parent = parent
        self.positives = positives
        self.negatives = negatives
        self.threshold = threshold


class DecisionTree:

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 3):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.not_used_features = []
        self.current_depth = 0

    def fit(self, X, y):
        self.not_used_features = list(X)

    def split(self, X, y):
        gini_per_feature = {}
        for feature in self.not_used_features:
            if X.dtypes[feature] == np.int64 or X.dtypes[feature] == np.float64:
                gini_pick_list = []
                X['y'] = y.tolist()
                sorted_data = X.sort_values(by=feature, ascending=True)
                feature_list = sorted_data[feature].tolist()
                feature_value_counts = X[feature].value_counts()
                n_total = sum(feature_value_counts)
                for i in range(len(feature_list) - 1):
                    p_sum = 0
                    median = (feature_list[i] + feature_list[i + 1]) / 2
                    less_than_median = sorted_data.loc[sorted_data[feature] < median, 'y'].value_counts()
                    n_less = sum(less_than_median)
                    for i in less_than_median.keys():
                        p_sum = p_sum + ((less_than_median[i] / n_less) ** 2)
                    gini_med_less = 1 - p_sum

                    p_sum = 0
                    bigger_than_median = sorted_data.loc[sorted_data[feature] >= median, 'y'].value_counts()
                    n_bigger = sum(bigger_than_median)
                    for i in bigger_than_median.keys():
                        p_sum = p_sum + ((bigger_than_median[i] / n_bigger) ** 2)
                    gini_med_bigger = 1 - p_sum

                    gini = (gini_med_less * (n_less / n_total)) + (gini_med_bigger * (n_bigger / n_total))
                    gini_pick_list.append(gini)
                gini_per_feature[feature] = min(gini_pick_list)
            else:
                feature_value_counts = X[feature].value_counts()
                n = sum(feature_value_counts)
                p_sum = 0
                for key in feature_value_counts.keys():
                    p_sum = p_sum + ((feature_value_counts[key] / n) ** 2)
                gini = 1 - p_sum
                gini_per_feature[feature] = gini
        print(gini_per_feature)

    def predict(self, X, y):
        pass
