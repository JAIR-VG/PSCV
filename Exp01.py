import sys
sys.path.append('src')  # o la ruta relativa a tu módulo
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import stratified_cv_by_class_and_group
from src import describe_protected_and_labels

# Suponiendo que ya tienes un BinaryLabelDataset `bld`
from aif360.datasets import AdultDataset


bld = AdultDataset(protected_attribute_names=['sex'],
                            privileged_classes=[['Male']],
                            features_to_keep=['age', 'education-num'])

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

describe_protected_and_labels(bld)


# Extraer etiquetas de clase
labels = bld.labels.ravel()
protected = bld.protected_attributes.ravel()

# Crear particiones con estratificación por clase
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X=np.zeros(len(labels)), y=labels))  # X puede ser dummy

# Visualizar distribución por clase en cada fold
for i, (train_idx, test_idx) in enumerate(folds):
    y_test = labels[test_idx]
    y_train = labels[train_idx]
    s_train = protected[train_idx]

    print(f"\nFold {i+1}")
    print(f"  Train size: {len(train_idx)}")
    print(f"  Test size:  {len(test_idx)}")

    from collections import Counter
    print(f"  Class distribution in train: {Counter(y_train)}")
    print(f"  Protected group distribution in train: {Counter(s_train)}")
    # Mostrar distribución conjunta (grupo, clase)
    df = pd.DataFrame({'group': s_train, 'label': y_train})
    group_label_counts = df.value_counts().sort_index()
    print(f"  Counts by (protected group, class):")
    print(group_label_counts)
    


    print(f"  Class distribution in test: {Counter(y_test)}")
    

# Obtener splits estratificados por clase y grupo sensible
folds = stratified_cv_by_class_and_group(bld, n_splits=5, random_state=42)

# Imprimir tamaños
for i, (train_idx, test_idx) in enumerate(folds):
    print(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
    y_train = labels[train_idx]
    s_train = protected[train_idx]

    # Mostrar distribución individual
    print(f"  Class distribution in train: {Counter(y_train)}")
    print(f"  Protected group distribution in train: {Counter(s_train)}")

    # Distribución conjunta (grupo, clase)
    df = pd.DataFrame({'group': s_train, 'label': y_train})
    counts = df.value_counts().sort_index()
    print(f"  Counts by (protected group, class):")
    print(counts)