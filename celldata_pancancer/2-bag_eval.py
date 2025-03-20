import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

print("###############################bicls########################################")
folds = [0]
for i in folds:
    print("split: ", i)
    data_csv = pd.read_csv(f"../eval_results/EVAL_bicls_pancancer_s1_cv/fold_{i}_test.csv")
    true_label = data_csv['Y']
    pred_label = data_csv['Y_hat']

    accuracy = accuracy_score(true_label, pred_label)
    auc = roc_auc_score(true_label, pred_label)
    macro_f1 = f1_score(true_label, pred_label, average="macro", zero_division=1)
    print("accuracy: ", round(accuracy,3))
    print("auc: ", round(auc,3))
    print("macro_f1: ", round(macro_f1,3))


print("###############################multicls########################################")
folds = [0]
for i in folds:
    print("split: ", i)
    df_bicls = pd.read_csv(f"../metrics_plot/data/fold_0_all.csv")
    df_multicls = pd.read_csv(f"../eval_results/EVAL_multicls_pancancer_s1_cv/fold_{i}_test.csv")
    
    df_duplicates = df_bicls[df_bicls['slide_id'].isin(df_multicls['slide_id'])]
    df_yhat_zero = df_duplicates[df_duplicates['Y_hat'] == 0]
    df_multicls_dup = df_multicls[df_multicls['slide_id'].isin(df_yhat_zero['slide_id'])]
    df_multicls.loc[df_multicls['slide_id'].isin(df_multicls_dup['slide_id']), 'Y_hat'] = 29.0
    
    true_label = df_multicls['Y']
    pred_label = df_multicls['Y_hat']

    accuracy = accuracy_score(true_label, pred_label)
    macro_f1 = f1_score(true_label, pred_label, average="macro", zero_division=1)
    print("accuracy: ", round(accuracy,3))
    print("macro_f1: ", round(macro_f1,3))