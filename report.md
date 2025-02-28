# Overview

The two relevant files for this project are `decision_tree.py` and `tree_training.ipynb`.

## Decision Tree

The DecisionTreeNode class defined in this file represents an implementation of both the Quinlan C4.5 and CART importance algorithms.

The constructor for my class accepts a string `algorithm` parameter to determine which importance method to use when creating the tree.

The class implements method for `build_decision_tree` which corresponds to the `fit` method of a typical sklearn model. It also implements `predict`.

The `build_decision_tree` method uses either the `select_lowest_gini_index` or the `select_highest_gain_ratio` to decide which attribute to split on.

Another method of interest is `compute_split_points` which determines the values to use when splitting the dataset on a continuous attribute.
Continuous attributes are identified explicitly by checking the column name against known continuous attributes for this dataset.

## Tree Training

The tree_training notebook performs the data fetching and preparation and the instantiation of DecisionTreeNodes. The notebook is well documented and results are discussed in there.

Overall, the CART algorithm performed better on the test set, despite having a lower 'best score' on the training set when compared to the C4.5 algorithm.

| Model | Dataset  | Score              |
| ----- | -------- | ------------------ |
| C4.5  | Training | 0.8571428571428572 |
| C4.5  | Test     | 0.7538461538461538 |
| CART  | Training | 0.84               |
| CART  | Test     | 0.7931034482758621 |

The CART algorithm had a noticably higher F1 score on the test set. This is somewhat counter-intuitive, since looking at the scores for the training data, output by the cross validation method, the C4.5 algorithm had consistently better predictions.

This suggests that C4.5 may be more prone to overfitting than the CART algorithm.
