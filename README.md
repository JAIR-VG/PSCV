# PSCV
Implements stratified cross-validation using both class labels and sensitive attributes. Ensures each fold maintains the distribution of (class × group) pairs, enabling fair and consistent evaluation of performance and fairness metrics. Compatible with AIF360's BinaryLabelDataset. 

Standard stratified cross-validation only preserves the distribution of class labels. However, when analyzing fairness, it’s crucial to ensure that each subgroup—like female class 0 or male class 1—is consistently represented across all folds. This method avoids biased metrics that may result from subgroup imbalance in test sets.

Each sample is assigned a composite label combining its class and sensitive attribute (e.g., "1_F"). StratifiedKFold is applied using these composite labels. You obtain train/test splits that maintain subgroup balance.
