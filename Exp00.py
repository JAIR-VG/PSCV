import sys
sys.path.append('src')  # o la ruta relativa a tu módulo
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src import paired_stratified_cv
from src import describe_protected_and_labels

# Cargando un BinaryLabelDataset
from aif360.datasets import AdultDataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing




dataset = AdultDataset(protected_attribute_names=['sex'],
                            privileged_classes=[['Male']],
                            features_to_keep=['age', 'education-num'])

col_name='sex'

privileged_groups = [{col_name: 1}]
unprivileged_groups = [{col_name: 0}]



# 2. Convertir a DataFrame
df, _ = dataset.convert_to_dataframe()
feature_names = dataset.feature_names
label_names = dataset.label_names[0]
protected_attr = dataset.protected_attribute_names[0]

# 3. Separar en arrays
X = df[feature_names].values
y = df[label_names].values
s = df[protected_attr].values  # 0 = Female, 1 = Male


describe_protected_and_labels(dataset)


# Extraer etiquetas de clase
#labels = bld.labels.ravel()
#protected = bld.protected_attributes.ravel()

# Crear particiones con estratificación por clase
folds = paired_stratified_cv(dataset, n_splits=5, random_state=42)


results = []

for i, (train_idx, test_idx) in enumerate(folds):
    y_test = y[test_idx]
    y_train = y[train_idx]
    s_train = s[train_idx]

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

    # Crear BinaryLabelDataset
    feature_cols = [f'x{i}' for i in range(X.shape[1])]
    df_train = pd.DataFrame(X[train_idx], columns=feature_cols)
    df_train['label'] = y[train_idx]
    df_train[col_name] = s[train_idx]

    dataset_orig_train = BinaryLabelDataset(df=df_train,
                                            label_names=['label'],
                                            protected_attribute_names=[col_name])
    
    metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    
    print("Reweighing")
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
    
    
    results.append({
        'fold': i + 1,
        'SPD_train_PSCV':metric_orig_train.statistical_parity_difference(),
        'DI_train_PSCV':metric_orig_train.disparate_impact(),
        'SPD_train_Reweighing_PSCV':metric_transf_train.statistical_parity_difference(),
        'DI_train_Reweighing_PSCV':metric_transf_train.disparate_impact()
    })




# Mostrar resultados
results_df = pd.DataFrame(results)
print("\nResumen de métricas por fold:")
print(results_df)
    