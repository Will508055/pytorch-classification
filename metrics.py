from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def calculate_metrics(model, x_test, y_test):
    y_pred_np = (model(x_test) > 0.5).detach().numpy()
    y_test_np = y_test.detach().numpy()

    roc_auc = roc_auc_score(y_test_np, y_pred_np)
    pr_auc = average_precision_score(y_test_np, y_pred_np)
    f1 = f1_score(y_test_np, y_pred_np)
    baseline_accuracy = y_test_np.mean()

    with open("results/metrics.txt", "a", encoding="utf-8") as f:
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}", file=f)
        print(f"ROC-AUC: {roc_auc:.4f}", file=f)
        print(f"PR-AUC: {pr_auc:.4f}", file=f)
        print(f"F1 Score: {f1:.4f}", file=f)