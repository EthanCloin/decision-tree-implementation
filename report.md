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

