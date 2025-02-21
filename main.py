import pandas as pd
import numpy as np
from decision_tree import DecisionTreeNode

"""
# data prep plan:
pull the training + test datasets from CreditApproval

fill missing values with training set median values
"""

"""
# decision tree plan:

identify each component of the data which must be considered by the decision tree
1. Attributes (A): column headers of data
2. Examples (X): rows in data
3. Labels (y): correct class for each example in the data
4. Dataset (D): collection of example + label - {(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)}
"""
node = DecisionTreeNode()

# 7 true, 3 false, expected entropy = 0.88129089
test_df = pd.DataFrame(
    {
        "Coffee Flavor": [
            "Caramel",
            "Caramel",
            "Caramel",
            "Caramel",
            "Caramel",
            "Caramel",
            "Caramel",
            "Cappucino",
            "Cappucino",
            "Cappucino",
        ],
        "Will Buy": [True, True, True, True, True, True, True, False, False, False],
    }
)
print(node.compute_entropy(test_df))
