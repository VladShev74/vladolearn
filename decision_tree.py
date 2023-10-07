class Node:
    def __init__(self, feature: str, left: Node, right: Node, parent: Node, positives: int, negatives: int,
                 threshold: float):
        self.feature = feature
        self.left = left
        self.right = right
        self.parent = parent
        self.positives = positives
        self.negatives = negatives
        self.threshold = threshold


class DecisionTree:

    def __init__(self, max_depth: int, min_samples_leaf: int):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.not_used_features = []
        self.current_depth = 0

    def fit(self, X, y):
        pass

    def split(self, X, y):
        pass

    def predict(self, X, y):
        pass