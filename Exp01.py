import sys
sys.path.append('src')  # o la ruta relativa a tu módulo
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import paired_stratified_cv
from src import describe_protected_and_labels

# Cargando un BinaryLabelDataset `bld`
from aif360.datasets import AdultDataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing



bld = AdultDataset(protected_attribute_names=['sex'],
                            privileged_classes=[['Male']],
                            features_to_keep=['age', 'education-num'])

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

df, _ = bld.convert_to_dataframe()
features = bld.feature_names
label_col = bld.label_names[0]
protected_attr = bld.protected_attribute_names[0]

X = df[features].values
y = df[label_col].values
s = df[protected_attr].values  # 0 = Female, 1 = Male


describe_protected_and_labels(bld)


# Extraer etiquetas de clase
labels = bld.labels.ravel()
protected = bld.protected_attributes.ravel()

# Crear particiones con estratificación por clase
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X=np.zeros(len(labels)), y=labels))  # X puede ser dummy


results = []
for i, (train_idx, test_idx) in enumerate(folds):
    X_train, y_train, s_train = X[train_idx], y[train_idx], s[train_idx]
    X_test, y_test, s_test = X[test_idx], y[test_idx], s[test_idx]
        # Dataset de entrenamiento (fairness estructural)
    dataset_orig_train = BinaryLabelDataset(df=pd.DataFrame({
        'label': y_train,
        'sex': s_train
    }), label_names=['label'], protected_attribute_names=['sex'])

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    print("SPD_train {:.6f}".format(metric_orig_train.statistical_parity_difference()))
    print("DI_train {:.6f}".format(metric_orig_train.disparate_impact()))

    print("Reweighing")
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
    print("SPD_train {:.6f}".format(metric_transf_train.statistical_parity_difference()))
    print("DI_train {:.6f}".format(metric_transf_train.disparate_impact()))

# Visualizar distribución por clase en cada fold
"""
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
"""

# Obtener splits estratificados por clase y grupo sensible

foldspaired = paired_stratified_cv(bld, n_splits=5, random_state=42)

# Imprimir tamaños
"""
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

"""
for i, (train_idx, test_idx) in enumerate(foldspaired):
    X_train, y_train, s_train = X[train_idx], y[train_idx], s[train_idx]
    X_test, y_test, s_test = X[test_idx], y[test_idx], s[test_idx]
        # Dataset de entrenamiento (fairness estructural)
    dataset_orig_train = BinaryLabelDataset(df=pd.DataFrame({
        'label': y_train,
        'sex': s_train
    }), label_names=['label'], protected_attribute_names=['sex'])

    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    print("PAIRED SPD_train {:.6f}".format(metric_orig_train.statistical_parity_difference()))
    print("PAIRED DI_train {:.6f}".format(metric_orig_train.disparate_impact()))

    print("Reweighing")
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
    print("SPD_train {:.6f}".format(metric_transf_train.statistical_parity_difference()))
    print("DI_train {:.6f}".format(metric_transf_train.disparate_impact()))
    