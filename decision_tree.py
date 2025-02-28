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

    def __init__(self, depth=0, max_depth=None, algorithm="C4.5"):
        # for tracking
        self.depth = depth
        self.max_depth = max_depth

        # for predicting
        self.children = {}
        self.attribute = None
        self.predicted_value = None
        self.backup_prediction = None

        # TODO: make this less hacky
        if algorithm == "C4.5":
            importance_method = self.select_highest_gain_ratio
        elif algorithm == "CART":
            importance_method = self.select_lowest_gini_index
        else:
            err = f"{algorithm} is invalid. accepted algorithms are 'C4.5' or 'CART'"
            raise ValueError(err)
        self.importance_method = importance_method
        self.algorithm = algorithm

        # for continuous
        self.threshold = None
        self.left = None
        self.right = None

    def print_tree(self, level=0):
        """Recursively prints the tree structure with indentation."""
        indent = "    " * level  # Indentation for readability

        if self.predicted_value is not None:  # Leaf node
            print(f"{indent}Predict: {self.predicted_value}")
            return

        if self.threshold is not None:  # Continuous split
            print(f"{indent}[{self.attribute} <= {self.threshold}]")
            if self.left:
                print(f"{indent}--> Left:")
                self.left.print_tree(level + 1)
            if self.right:
                print(f"{indent}--> Right:")
                self.right.print_tree(level + 1)
        else:  # Discrete split
            print(f"{indent}[Split on {self.attribute}]")
            for value, child in self.children.items():
                print(f"{indent}--> {self.attribute} = {value}:")
                child.print_tree(level + 1)

    def build_decision_tree(self, dataset: pd.DataFrame, parent_examples=None):
        """returns a node"""
        # base cases

        # no more examples
        if dataset.empty:
            self.predicted_value = self.plurality_value(parent_examples)
            return self
        # only label column remains
        if len(dataset.columns) == 1:
            self.predicted_value = self.plurality_value(dataset)
            return self

        # hit my max depth
        if self.depth == self.max_depth:
            self.predicted_value = self.plurality_value(dataset)
            return self

        labels = dataset.iloc[:, -1]
        # all same label
        if labels.nunique() == 1:
            self.predicted_value = self.plurality_value(dataset)
            return self

        # saving plurality value for each node as a default in case of missing children
        self.backup_prediction = self.plurality_value(dataset)

        # TODO: might need to handle if attribute ever comes back as None
        attribute, best_split = self.importance_method(dataset)
        self.attribute = attribute

        if best_split is not None:  # split on continuous attr case
            self.threshold = best_split
            self.left = DecisionTreeNode(
                depth=self.depth + 1, max_depth=self.max_depth, algorithm=self.algorithm
            )
            self.right = DecisionTreeNode(
                depth=self.depth + 1, max_depth=self.max_depth, algorithm=self.algorithm
            )

            self.left.build_decision_tree(
                dataset[dataset[attribute] <= self.threshold], dataset
            )
            self.right.build_decision_tree(
                dataset[dataset[attribute] > self.threshold], dataset
            )
        else:  # split on categorical attr case
            for unique_value in np.unique(dataset[attribute]):
                matching_examples = dataset[dataset[attribute] == unique_value].copy()

                # drop attribute, same idea as passing in attributes - A in pseudocode
                matching_examples = matching_examples.drop(columns=[attribute])
                child = DecisionTreeNode(
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    algorithm=self.algorithm,
                )
                self.children[unique_value] = child
                child.build_decision_tree(matching_examples, dataset)
        return self

    def predict(self, dataset):
        if self.threshold is None and self.children is None:
            raise Exception("tree must be built before predicting")
        return dataset.apply(lambda example: self._predict_example(example), axis=1)

    def _predict_example(self, example):
        """
        in this case my dataset has no labels and my return value should be
        a Series of same len as dataset but it's the predicted label values
        """
        # first navigate to a leaf node
        node = self
        while node.predicted_value is None:
            if node.threshold is not None:  # continuous split
                if example[node.attribute] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:  # categorical split
                if example[node.attribute] in node.children:
                    node = node.children[example[node.attribute]]
                else:
                    # this is a weird case. i would like to return the plurality value
                    # here but i don't have access to it?
                    # update tree building to save it as an attr on the node?
                    return node.backup_prediction

        return node.predicted_value

    def plurality_value(self, dataset: pd.DataFrame):
        return dataset.iloc[:, -1].mode().values[0]

    def select_lowest_gini_index(self, dataset):
        examples = dataset.iloc[:, 0:-1]

        lowest_gini = float("inf")
        selected_attr = None
        best_split = None

        for attribute in examples.columns:
            gini, split_point = self.compute_gini_index(dataset, attribute)
            if gini < lowest_gini:
                lowest_gini = gini
                selected_attr = attribute
                best_split = split_point

        return selected_attr, best_split

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
                    best_split = t
            return best_gini, best_split
        else:
            gini_index = 0
            for unique_value in np.unique(dataset[attribute]):
                matching_examples = dataset[dataset[attribute] == unique_value]
                gini_value = self.compute_gini_value(matching_examples)
                partial_gini_index = (
                    len(matching_examples) / len(dataset)
                ) * gini_value
                gini_index += partial_gini_index
            return gini_index, None

    def compute_split_points(self, dataset, attribute):
        split_points = []

        attr_values = np.sort(np.unique(dataset[attribute]))
        # compute split points
        for i, val in enumerate(attr_values):
            # avoid outofbounds err
            if i == len(attr_values) - 1:
                continue
            next_val = attr_values[i + 1]
            midpoint = (val + next_val) / 2
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
            gain, split_point = self.compute_gain_ratio(dataset, attribute)
            if gain > highest_gain:
                highest_gain = gain
                selected_attr = attribute
                best_split = split_point
        return selected_attr, best_split

    def compute_gain_ratio(self, dataset, attribute):
        """returns gain ratio and best split value if attribute is continuous"""
        # BUG: A9 is a categorical attribute but returning a best_split value?
        gain, best_split = self.compute_information_gain(dataset, attribute)
        iv = self.compute_intrinsic_value(dataset, attribute, best_split)
        gain_ratio = gain / iv if iv != 0 else 0
        return gain_ratio, best_split

    def compute_intrinsic_value(self, dataset, attribute, split_point=None):
        if split_point:
            left = dataset[dataset[attribute] <= split_point]
            right = dataset[dataset[attribute] > split_point]

            left_weight = len(left) / len(dataset)
            right_weight = len(right) / len(dataset)

            intrinsic_value = -(
                (left_weight * np.log2(left_weight) if left_weight > 0 else 0)
                + (right_weight * np.log2(right_weight) if right_weight > 0 else 0)
            )
            return intrinsic_value
        else:
            intrinsic_value = 0
            for unique_value in np.unique(dataset[attribute]):
                matching_examples = dataset[dataset[attribute] == unique_value]
                weight = len(matching_examples) / len(dataset)
                partial_iv = weight * np.log2(weight) if weight > 0 else 0

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
            return best_gain, best_split
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
            return information_gain, None

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
        """as defined in the dataset description"""
        continuous_attributes = ["A15", "A14", "A11", "A8", "A3", "A2"]
        return attribute in continuous_attributes
