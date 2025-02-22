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
# TODO: move these into a unit test folder
node = DecisionTreeNode()

# 7 true, 3 false, expected entropy = 0.88129089
# test_df = pd.DataFrame(
#     {
#         "Coffee Flavor": [
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Cappucino",
#             "Cappucino",
#             "Cappucino",
#         ],
#         "Will Buy": [True, True, True, True, True, True, True, False, False, False],
#     }
# )
# print(node.compute_entropy(test_df))

# test_df = pd.DataFrame(
#     {
#         "Coffee Flavor": [
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Caramel",
#             "Cappucino",
#             "Cappucino",
#             "Cappucino",
#         ],
#         "Will Buy": [True, True, True, True, True, True, True, True, True, True],
#     }
# )
simple_example = pd.DataFrame(
    {
        "X": [1, 1, 0, 1],
        "Y": [1, 1, 0, 0],
        "Z": [1, 0, 1, 0],
        "C": ["I", "I", "II", "II"],
    }
)
# expect information_gain 0.31127812445913283
print(node.compute_information_gain(simple_example, "X"))
# expect information_gain 1.0
print(node.compute_information_gain(simple_example, "Y"))
# expect information_gain 0
print(node.compute_information_gain(simple_example, "Z"))
# expect intrinsic value 0.8112781244591328
print(node.compute_intrinsic_value(simple_example, "X"))
# expect gain ratio 0.3836885465963443
print(node.compute_gain_ratio(simple_example, "X"))
