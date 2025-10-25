import numpy as np
import pandas as pd


#The Math stuff:

def sigmoid(z):
    # We clip z to narrow down its range: ir might go crazy and end up with weird numbers
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_linear_score(X, weights, bias):
    return np.dot(X, weights) + bias


def predict_probability(X, weights, bias):
    z = compute_linear_score(X, weights, bias)
    return sigmoid(z)


def compute_log_likelihood(X, y, weights, bias):
    m = X.shape[0]
    y_pred = predict_probability(X, weights, bias)

    # We ofc add a 1e-15 to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    log_likelihood = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return log_likelihood / m


def compute_gradients(X, y, weights, bias):
    m = X.shape[0]
    y_pred = predict_probability(X, weights, bias)
    error = y_pred - y

    dw = (1 / m) * np.dot(X.T, error)
    db = (1 / m) * np.sum(error)

    return dw, db


def train_model(X, y, learning_rate=0.1, n_iterations=500, verbose=True):
    m, n = X.shape

    # here we start with some random guesses
    weights = np.random.randn(n) * 0.01
    bias = 0.0
    cost_history = []

    for i in range(n_iterations):
        # this makes us see how we're doing
        current_log_likelihood = compute_log_likelihood(X, y, weights, bias)
        cost_history.append(current_log_likelihood)

        # then we figure out which way to move according to the algorithm
        dw, db = compute_gradients(X, y, weights, bias)

        # Take a step
        weights += learning_rate * dw
        bias += learning_rate * db

        if verbose and (i % 100 == 0 or i == n_iterations - 1):
            print(f"Step {i}: Score = {current_log_likelihood:.4f}")

    return weights, bias, cost_history


def predict(X, weights, bias, threshold=0.5):
    probabilities = predict_probability(X, weights, bias)
    return (probabilities >= threshold).astype(int)


def compute_odds_ratio(weights, feature_index):
    return np.exp(weights[feature_index])


# Data handling:

def clean_data(df):
    df = df.copy()

    # Fixing those TotalCharges, in the case of an error
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)

    # We convert Yes/No to 1/0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    return df


def one_hot_encode_column(df, column):
    if column not in df.columns:
        return df

    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    result = pd.concat([df.drop(column, axis=1), dummies], axis=1)


    if column in result.columns:
        result.drop(column, axis=1, inplace=True)

    return result


def scale_features(df, features):
    df = df.copy()

    for feature in features:
        if feature in df.columns:
            mean_val = df[feature].mean()
            std_val = df[feature].std()

            if std_val == 0:
                std_val = 1e-10 # We avoid division by zero
            df[feature] = (df[feature] - mean_val) / std_val

    return df


def prepare_data(df):
    df = clean_data(df)

    # we convert tje categorical stuff
    categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
    for col in categorical_cols:
        df = one_hot_encode_column(df, col)

    # we scale the numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df = scale_features(df, numerical_cols)

    # Split features from target
    if 'Churn' in df.columns:
        y = df['Churn'].values
        X = df.drop('Churn', axis=1)
    else:
        y = None
        X = df

    # we're only interested in the numbers
    X = X.select_dtypes(include='number')

    return X.values, y


#Evaluation Stuff:

def confusion_matrix(true_labels, predicted_labels):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    total_customers = len(true_labels)

    for i in range(total_customers):
        actual_result = true_labels[i]
        predicted_result = predicted_labels[i]

        if actual_result == 1 and predicted_result == 1:
            true_positive += 1
        elif actual_result == 0 and predicted_result == 0:
            true_negative += 1
        elif actual_result == 0 and predicted_result == 1:
            false_positive += 1
        elif actual_result == 1 and predicted_result == 0:
            false_negative += 1

    return true_positive, true_negative, false_positive, false_negative


def precision(tp, fp):

    return tp / (tp + fp + 1e-15)


def recall(tp, fn):

    return tp / (tp + fn + 1e-15)


def accuracy(tp, tn, fp, fn):

    return (tp + tn) / ( tp + tn + fp + fn  + 1e-15)


def f1_score(precision_val, recall_val):
    denominator = precision_val + recall_val
    return 2 * (precision_val * recall_val) / denominator if denominator > 0 else 0.0


def compute_roc_auc(true_labels, prediction_scores):
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []

    for threshold in thresholds:
        predictions = (prediction_scores >= threshold).astype(int)

        tp = tn = fp = fn = 0
        for actual, predicted in zip(true_labels, predictions):
            if actual == 1 and predicted == 1:
                tp += 1
            elif actual == 0 and predicted == 0:
                tn += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tprs.append(tpr)
        fprs.append(fpr)

    # we finish it off by calculating the area under curve ( auc) the simple way
    auc = 0.0
    for i in range(1, len(fprs)):
        width = fprs[i] - fprs[i - 1]
        avg_height = (tprs[i] + tprs[i - 1]) / 2
        auc += width * avg_height

    return auc


def evaluate_model(true_labels, predicted_labels, prediction_scores):
    tp, tn, fp, fn = confusion_matrix(true_labels, predicted_labels)

    precision_val = precision(tp, fp)
    recall_val = recall(tp, fn)
    accuracy_val = accuracy(tp, tn, fp, fn)
    f1_val = f1_score(precision_val, recall_val)
    auc_val = compute_roc_auc(true_labels, prediction_scores)

    return tp, tn, fp, fn, precision_val, recall_val, accuracy_val, f1_val, auc_val


def train_test_split(features, labels, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)

    n_samples = features.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    return X_train, X_test, y_train, y_test


# Let's run this thing:

def run_complete_example():
    print("LOGISTIC REGRESSION - TELCO CHURN PREDICTION")
    print("=" * 10)

    # We start off by making some fake data that looks real
    print("\nCreating sample data.")
    np.random.seed(42)
    n_customers = 1000

    sample_data = pd.DataFrame({
        'tenure': np.random.randint(1, 73, n_customers),
        'MonthlyCharges': np.random.uniform(20, 120, n_customers),
        'TotalCharges': np.random.uniform(100, 8000, n_customers),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'Churn': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7])
    })

    print("🛠Preparing data:")
    X, y = prepare_data(sample_data)
    print(f"   Dataset: {X.shape[0]} customers, {X.shape[1]} features")
    print(f"   Churn rate: {np.mean(y):.1%}")

    print("Splitting data:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")

    print("\nTraining model.")
    weights, bias, history = train_model(X_train, y_train, learning_rate=0.1, n_iterations=500, verbose=True)

    print("\nMaking predictions.")
    y_pred = predict(X_test, weights, bias)
    y_probs = predict_probability(X_test, weights, bias)

    print("\nResults:")
    print("-" * 10)
    tp, tn, fp, fn, prec, rec, acc, f1, auc = evaluate_model(y_test, y_pred, y_probs)

    print(f"\nConfusion Matrix:")
    print(f"               | Actual Churn | Actual Stay")
    print(f"Predict Churn  |     {tp:4d}     |     {fp:4d}")
    print(f"Predict Stay   |     {fn:4d}     |     {tn:4d}")

    print(f"\nThe Metrics:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  AUC:       {auc:.3f}")

    print(f"\n The feature insights (Top 5):")
    print("-" * 30)
    for i in range(min(5, len(weights))):
        odds_ratio = compute_odds_ratio(weights, i)
        if odds_ratio > 1:
            direction = "increases"
            change_pct = (odds_ratio - 1) * 100
        else:
            direction = "decreases"
            change_pct = (1 - odds_ratio) * 100

        print(f"  Feature {i}: OR = {odds_ratio:.3f} ({direction} odds by {change_pct:.1f}%)")

    print("\n ")
    return weights, bias


if __name__ == "__main__":
    final_weights, final_bias = run_complete_example()