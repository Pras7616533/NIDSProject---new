import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(y):
    """
    Compute class weights for imbalanced dataset

    Parameters:
    y (array-like): Target labels

    Returns:
    class_weights (dict): Class weight mapping
    """

    print("\nComputing class weights...")

    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )

    class_weights = dict(zip(classes, weights))

    print("Class weights computed:")
    print(class_weights)

    return class_weights
