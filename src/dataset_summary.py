import numpy as np
import pandas as pd
from collections import Counter
from aif360.datasets import BinaryLabelDataset

def describe_protected_and_labels(dataset: BinaryLabelDataset):
    """
    Prints and returns the count of instances by protected group and class label.

    Parameters:
    - dataset: BinaryLabelDataset

    Returns:
    - pandas.Series with counts by (protected group, class)
    """
    protected = dataset.protected_attributes.ravel()
    labels = dataset.labels.ravel()
    
    # 1. Valores únicos del atributo protegido
    unique_protected_values = np.unique(protected)
    print(f"Protected attribute values: {unique_protected_values}")
    print(f"Number of protected groups: {len(unique_protected_values)}")
    
    # 2. Valores únicos de la clase
    unique_labels = np.unique(labels)
    print(f"Class labels: {unique_labels}")
    print(f"Number of classes: {len(unique_labels)}")
    
    # 3. Conteo por combinación
    df = pd.DataFrame({'group': protected, 'label': labels})
    counts = df.value_counts().sort_index()
    print("\nCounts by (protected group, class):")
    print(counts)

    return counts
