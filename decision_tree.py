import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional


class Node:

    def __init__(self,
                 feature: str,
                 less_median: list,
                 bigger_median: list,
                 threshold: float,
                 gini: float,
                 left: Optional['Node'] = None,
                 right: Optional['Node'] = None,
                 parent: Optional['Node'] = None):
        self.feature = feature
        self.gini = gini
        self.left = left
        self.right = right
        self.parent = parent
        self.less_median = less_median
        self.bigger_median = bigger_median
        self.threshold = threshold
        self.is_leaf = False

    def __repr__(self):
        return f'Node(feature={self.feature}, threshold={self.threshold}, gini={self.gini})'


class DecisionTree:

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 3):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.not_used_features = []
        self.current_depth = 0
        self.root = None

    def fit(self, X, y):
        self.not_used_features = list(X)

    def split(self, X, y):

        gini_per_feature = {}
        for feature in self.not_used_features:

            if np.issubdtype(X.dtypes[feature], np.number):
                X['y'] = y.tolist()
                sorted_data = X.sort_values(by=feature, ascending=True)
                distinct_values_list = sorted_data[feature].unique().tolist()
                total_distinct_values = sum(X[feature].value_counts())

                for idx in range(len(distinct_values_list) - 1):
                    p_sum_less = 0
                    median = (distinct_values_list[idx] + distinct_values_list[idx + 1]) / 2

                    # print(median)
                    # print(distinct_values_list)
                    # print(feature)

                    less_than_median = sorted_data.loc[sorted_data[feature] < median, 'y'].value_counts()
                    values_less = sum(less_than_median)

                    for label in less_than_median.keys():
                        p_sum_less = p_sum_less + ((less_than_median[label] / values_less) ** 2)
                    gini_med_less = 1 - p_sum_less

                    p_sum_bigger = 0
                    bigger_than_median = sorted_data.loc[sorted_data[feature] >= median, 'y'].value_counts()
                    values_bigger = sum(bigger_than_median)

                    for i in bigger_than_median.keys():
                        p_sum_bigger = p_sum_bigger + ((bigger_than_median[i] / values_bigger) ** 2)
                    gini_med_bigger = 1 - p_sum_bigger

                    gini = (gini_med_less * (values_less / total_distinct_values)) + (
                                gini_med_bigger * (values_bigger / total_distinct_values))
                    values_before_median = []
                    values_after_median = []
                    for class_label in less_than_median.keys():
                        values_before_median += [class_label] * less_than_median[class_label]
                    for class_label in bigger_than_median.keys():
                        values_after_median += [class_label] * bigger_than_median[class_label]

                    if feature not in gini_per_feature.keys():
                        gini_per_feature[feature] = (gini, median, values_before_median, values_after_median)
                    else:
                        if gini < gini_per_feature[feature][0]:
                            gini_per_feature[feature] = (gini, median, values_before_median, values_after_median)

            else:
                raise ValueError('Ti che suka ahuel, davai mne number')

        gini_per_feature = dict(sorted(gini_per_feature.items(), key=lambda item: item[1]))

        for feature in self.not_used_features:
            current_best_feature = next(iter(gini_per_feature))
            # print(current_best_feature)
            # print(best_feature_threshold)

            if self.root is None:
                self.root = Node(
                    feature=current_best_feature,
                    less_median=gini_per_feature[current_best_feature][2],
                    bigger_median=gini_per_feature[current_best_feature][3],
                    threshold=gini_per_feature[current_best_feature][1],
                    gini=gini_per_feature[current_best_feature][0]
                )

            print(self.root)
            break

            gini_per_feature.pop(current_best_feature)

    def predict(self, X, y):
        pass
