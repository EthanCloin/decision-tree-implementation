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

    # TODO: figure out bipartition stuff and consider if i need to label attributes as continuous

    def __init__(self):
        self.children = {}
        self.attribute = None
        self.predicted_value = None

    def build_decision_tree(self, examples, attributes, parent_examples):
        pass

    def select_lowest_gini_index(self, dataset):
        examples = dataset.iloc[:, 0:-1]

        lowest_gini = float("inf")
        selected_attr = None
        for attribute in examples.columns:
            gini = self.compute_gini_index(dataset, attribute)
            if gini < lowest_gini:
                lowest_gini = gini
                selected_attr = attribute
        return selected_attr

    # TODO: get a test value to dummy check this
    # NOTE: this one checks if the attr is continuous and decides to do bipartition if needed
    def compute_gini_index(self, dataset, attribute):
        if self.is_continuous(attribute):
            best_gini = float("inf")
            best_split = None
            split_points = self.compute_split_points(dataset, attribute)

            # compute gini index for each split pint
            for t in split_points:
                left = dataset[dataset[attribute] <= t]
                right = dataset[dataset[attribute] > t]

                left_gini = self.compute_gini_value(left)
                right_gini = self.compute_gini_value(right)
                weighted_gini = ((len(left) / len(dataset)) * left_gini) + (
                    (len(right) / len(dataset)) * right_gini
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = t  # not returning now but might need to add to the node or smth?
            return best_gini
        else:
            gini_index = 0
            for unique_value in np.unique(dataset[attribute]):
                matching_examples = dataset[dataset[attribute] == unique_value]
                gini_value = self.compute_gini_value(matching_examples)
                partial_gini_index = (
                    len(matching_examples) / len(dataset)
                ) * gini_value
                gini_index += partial_gini_index
            return gini_index

    def compute_split_points(self, dataset, attribute):
        split_points = []

        attr_values = np.sort(np.unique(dataset[attribute]))
        # compute split points
        for val, i in enumerate(attr_values):
            # avoid outofbounds err
            if i == len(attr_values) - 1:
                continue
            midpoint = (val + attr_values[i + 1]) / 2
            split_points.append(midpoint)
        return split_points

    def compute_gini_value(self, dataset):
        # 1 - sum(y in unique_labels p_k^2)
        # where p_k is the percent of examples in D labeled by y
        gini_value = 0
        # assuming last col in dataset is labels
        labels = dataset.iloc[:, -1]
        for y in np.unique(labels):
            p_k = len([label for label in labels if label == y]) / len(labels)
            gini_value += p_k * p_k
        return 1 - gini_value

    def select_highest_gain_ratio(self, dataset: pd.DataFrame):
        """this is the Importance fxn for Quinlan C4.5"""
        # assuming last col in dataset is labels
        examples = dataset.iloc[:, 0:-1]

        highest_gain = 0
        selected_attr = None
        for attribute in examples.columns:
            gain = self.compute_gain_ratio(dataset, attribute)
            if gain > highest_gain:
                highest_gain = gain
                selected_attr = attribute
        return selected_attr

    def compute_gain_ratio(self, dataset, attribute):
        gain = self.compute_information_gain(dataset, attribute)
        iv = self.compute_intrinsic_value(dataset, attribute)
        return gain / iv

    def compute_intrinsic_value(self, dataset, attribute):
        intrinsic_value = 0
        for unique_value in np.unique(dataset[attribute]):
            matching_examples = dataset[dataset[attribute] == unique_value]
            partial_iv = (len(matching_examples) / len(dataset)) * np.log2(
                (len(matching_examples) / len(dataset))
            )
            intrinsic_value += partial_iv
        return -intrinsic_value

    def compute_information_gain(self, dataset: pd.DataFrame, attribute: str):
        if self.is_continuous(attribute):
            split_points = self.compute_split_points(dataset, attribute)
            parent_entropy = self.compute_entropy(dataset)
            best_gain = float("-inf")
            best_split = None

            for t in split_points:
                left = dataset[dataset[attribute] <= t]
                right = dataset[dataset[attribute] > t]

                left_entropy = self.compute_entropy(left)
                right_entropy = self.compute_entropy(right)
                weighted_entropy = ((len(left) / len(dataset)) * left_entropy) + (
                    (len(right) / len(dataset)) * right_entropy
                )
                gain = parent_entropy - weighted_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_split = t
            return best_gain
        else:
            # information gain measures the benefit of splitting on a particular attribute
            # Gain(D, a) = Entropy(parent) –[weighted_average Entropy(children)]
            parent_entropy = self.compute_entropy(dataset)
            information_gain = parent_entropy

            for unique_value in np.unique(dataset[attribute]):
                matching_examples = dataset[dataset[attribute] == unique_value]
                attr_entropy = self.compute_entropy(matching_examples)
                weighted_entropy = (
                    len(matching_examples) / len(dataset)
                ) * attr_entropy
                information_gain -= weighted_entropy
            return information_gain

    def compute_entropy(self, dataset: pd.DataFrame):
        # entropy(D) = -sum(each k in K p_k * log_2(p_k))
        # where p_k is percent of examples in D labeled by y_k

        # assuming last col in dataset is labels
        labels = dataset.iloc[:, -1]
        entropy = 0.0
        for y in np.unique(labels):
            # percent of examples which have current label
            p_k = len([label for label in labels if label == y]) / len(labels)
            entropy += p_k * np.log2(p_k)
        return 0 if entropy == 0 else -entropy

    def is_continuous(self, attribute):
        continuous_attributes = ["A15", "A14", "A11", "A8", "A3", "A2"]
        return attribute in continuous_attributes
