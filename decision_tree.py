"""
notated pseudocode from slides:

function Decision-Tree-Learning(examples, attributes, parent_examples) -> Tree
    if examples is empty
        # what do we mean 'Plurality-Value'?
        #   the most common classification, so if i have 10 parent examples
        #   where 3 are True and 7 are False, return False
        return Plurality-Value(parent_examples)
    else if all examples have the same classification
        return classification
    else if attributes is empty
        return Plurality-Value(examples)
    else
        A = max(Importance(a, examples) for a in attributes)
        tree = new Tree(root=A)
        # what do we mean "value of A"? 
        #   valid instances of attribute A like "Y,N" for categorical
        #   or bipartition for continuous
        for each possible_val of A
            matching_examples = e for e in examples if e.A = possible_val
            subtree = Decision-Tree-Learning(matching_examples, attributes - A, examples)
"""

import numpy as np
import pandas as pd


class DecisionTreeNode:
    """
    the point of this node class is to compose a decision tree for a binary classification problem.
    important attributes of my node class include

    - predicted_value: this will have a value only for leaf nodes
    - attribute: which column in the dataset the node is basing its decision on
    - children: dictionary mapping to nodes deeper in the tree
    """

    def __init__(self):
        self.children = {}
        self.attribute = None
        self.predicted_value = None

    def build_decision_tree(self, examples, attributes, parent_examples):
        pass

    def compute_information_gain(self, dataset: pd.DataFrame, attribute: str):
        # information gain measures the benefit of splitting on a particular attribute
        # Gain(D, a) = Entropy(parent) –[weighted_average Entropy(children)]
        parent_entropy = self.compute_entropy(dataset)
        information_gain = parent_entropy

        for attr in np.unique(dataset[attribute]):
            matching_examples = dataset[dataset[attribute] == attr]
            attr_entropy = self.compute_entropy(matching_examples)
            weighted_entropy = (len(matching_examples) / len(dataset)) * attr_entropy
            information_gain -= weighted_entropy
        return information_gain

    def compute_entropy(self, dataset: pd.DataFrame):
        # entropy(D) = -sum(each k in K p_k * log_2(p_k))
        # where p_k is percent of examples in D labeled by y_k
        labels = dataset.iloc[:, -1]
        entropy = 0.0
        for y in np.unique(labels):
            # percent of examples which have current label
            p_k = len([label for label in labels if label == y]) / len(labels)
            entropy += p_k * np.log2(p_k)
        return 0 if entropy == 0 else -entropy
