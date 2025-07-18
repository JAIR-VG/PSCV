import sys
sys.path.append('src')  # o la ruta relativa a tu módulo
import numpy as np
import pandas as pd
from src import paired_stratified_cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from aif360.algorithms.preprocessing import Reweighing



# Cargando un BinaryLabelDataset 
from aif360.datasets import AdultDataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric


dataset = AdultDataset(protected_attribute_names=['sex'],
                            privileged_classes=[['Male']],
                            features_to_keep=['age', 'education-num'])

privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]

# 2. Convertir a DataFrame
df, _ = dataset.convert_to_dataframe()
feature_names = dataset.feature_names
label_names = dataset.label_names[0]
protected_attr = dataset.protected_attribute_names[0]

# 3. Separar en arrays
X = df[feature_names].values
y = df[label_names].values
s = df[protected_attr].values  # 0 = Female, 1 = Male


# Crear particiones con estratificación por clase
folds = paired_stratified_cv(dataset, n_splits=5, random_state=42)


results = []
for i, (train_idx, test_idx) in enumerate(folds):
    #6.1 Crear DataFrames para train y test con todas las columnas
    feature_cols = [f'x{i}' for i in range(X.shape[1])]

    df_train = pd.DataFrame(X[train_idx], columns=feature_cols)
    df_train['label'] = y[train_idx]
    df_train['sex'] = s[train_idx]

    df_test = pd.DataFrame(X[test_idx], columns=feature_cols)
    df_test['label'] = y[test_idx]
    df_test['sex'] = s[test_idx]



     # Crear BinaryLabelDataset
    dataset_orig_train = BinaryLabelDataset(df=df_train,
                                            label_names=['label'],
                                            protected_attribute_names=['sex'])

    dataset_orig_test = BinaryLabelDataset(df=df_test,
                                           label_names=['label'],
                                           protected_attribute_names=['sex'])

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(dataset_orig_train)

    X_train = dataset_transf_train.features
    y_train = dataset_transf_train.labels.ravel()
    w_train = dataset_transf_train.instance_weights

        # Entrenar
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train,sample_weight=w_train)
    #clf.fit(X_train, y_train)

    # Evaluar en test
    X_test = dataset_orig_test.features
    y_test = dataset_orig_test.labels.ravel()

    y_pred = clf.predict(X_test)
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Recall por clase (ejemplo para clase '1.0')
    recall_clase_1 = report['1.0']['recall']  # si las etiquetas son floats
    recall_clase_0 = report['0.0']['recall']  # si las etiquetas son floats

    #Evaluar fairness
    dataset_pred = dataset_orig_test.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)

    metric = ClassificationMetric(dataset_orig_test, dataset_pred,
                                  privileged_groups=privileged_groups,
                                  unprivileged_groups=unprivileged_groups)


    results.append({
        'fold': i + 1,
        'accuracy': acc,
        'Recall_0': recall_clase_0,
        'Recall_1': recall_clase_1,
        'EOD': metric.equal_opportunity_difference(),
        'AOD': metric.average_odds_difference(),
        'ERR': metric.error_rate_difference(),
        'FDRD': metric.false_discovery_rate_difference(),
        'FNRD': metric.false_negative_rate_difference()
    })

# Mostrar resultados
results_df = pd.DataFrame(results)
print("\nResumen de métricas por fold:")
print(results_df)
print(f"\nPromedio de accuracy: {results_df['accuracy'].mean():.4f}")
