import numpy as np
from sklearn.model_selection import StratifiedKFold
from aif360.datasets import BinaryLabelDataset

def stratified_cv_by_class_and_group(dataset: BinaryLabelDataset, n_splits=5, shuffle=True, random_state=None):
    """
    Performs stratified cross-validation by class and sensitive attribute pairs.

    Parameters:
    - dataset: BinaryLabelDataset from AIF360
    - n_splits: number of folds
    - shuffle: whether to shuffle before splitting
    - random_state: seed for reproducibility

    Returns:
    - List of (train_index, test_index) tuples for each fold
    """
    labels = dataset.labels.ravel()
    protected_attr = dataset.protected_attributes.ravel()
    
    combo = [f"{y}_{s}" for y, s in zip(labels, protected_attr)]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    splits = list(skf.split(X=np.zeros(len(combo)), y=combo))
    
    return splits
