import matplotlib.pyplot as plt
import Hypothesis1.Hypothesis1 as H1
import Hypothesis2.DataExplosure as H2_1
import Hypothesis2.ExposureIndustry as H2_2
import Hypothesis3.kaggle as H3_kaggle
import Hypothesis3.WA as H3_WA
import Hypothesis4.DecisionTree as H4_DecisionTree
import Hypothesis4.Visualization as H4_Visualization
import sys
import os
import warnings
from contextlib import contextmanager

@contextmanager
def suppress_show():
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        yield
    finally:
        plt.show = original_show


@contextmanager
def suppress_print_and_warnings():
    # Save original stdout
    original_stdout = sys.stdout
    # Redirect stdout to null
    sys.stdout = open(os.devnull, 'w')
    # Suppress warnings
    warnings.filterwarnings('ignore')
    try:
        yield
    finally:
        # Restore stdout and warnings after block
        sys.stdout.close()
        sys.stdout = original_stdout
        warnings.resetwarnings()

with suppress_show():
    # Run all hypothesis scripts
    print("Running Hypothesis 1...")
    with suppress_print_and_warnings():
        H1.main()
    print("Running Hypothesis 2 - Data Exposure...")
    with suppress_print_and_warnings():
        H2_1.main()
    print("Running Hypothesis 2 - Exposure Industry...")
    with suppress_print_and_warnings():
        H2_2.main()
    print("Running Hypothesis 3 - Kaggle Dataset...")
    with suppress_print_and_warnings():
        H3_kaggle.main()
    print("Running Hypothesis 3 - Washington Dataset...")
    with suppress_print_and_warnings():
        H3_WA.main()
    print("Running Hypothesis 4 - Decision Tree...")
    with suppress_print_and_warnings():
        H4_DecisionTree.main()
    print("Running Hypothesis 4 - Visualization...")
    with suppress_print_and_warnings():
        H4_Visualization.main()
