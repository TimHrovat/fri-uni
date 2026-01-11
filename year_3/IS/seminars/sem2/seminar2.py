import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from IPython.display import display, HTML


sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_english_dataset():
    train_data = load_jsonl('data/English dataset/train.jsonl')
    test_data = load_jsonl('data/English dataset/test.jsonl')

    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def t1_transform_to_binary(df, strategy='contradiction_vs_rest'):
    """Transform multi-class labels to binary classification."""
    df = df.copy()

    if strategy == 'contradiction_vs_rest':
        df['binary_label'] = (df['label'] == 'Contradiction').astype(int)

    return df


def t1_combine_text_pairs(df):
    """Combine premise and hypothesis into text pairs for analysis."""
    df = df.copy()

    df['text_pair'] = df['premise'] + ' [SEP] ' + df['hypothesis']
    df['premise_filled'] = df['premise'].replace('', '[NO PREMISE]')

    return df



def t1_check_missing_values(df):
    """Check for missing values and empty strings in the dataset."""
    text_cols = ['premise', 'hypothesis']
    quality_data = []

    for col in text_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            empty_count = (df[col] == '').sum()
            total_issues = null_count + empty_count
            quality_data.append({
                'Column': col,
                'Null Values': null_count,
                'Empty Strings': empty_count,
                'Total Issues': total_issues,
                'Issue Rate (%)': f'{(total_issues / len(df) * 100):.2f}%'
            })

    return pd.DataFrame(quality_data)


def t1_check_duplicates(df):
    """Check for duplicate entries in the dataset."""
    duplicates = df.duplicated().sum()
    text_pair_dupes = df.duplicated(subset=['premise', 'hypothesis']).sum()
    premise_dupes = df[df['premise'] != ''].duplicated(subset=['premise']).sum()

    dup_data = {
        'Duplicate Type': [
            'Exact duplicate rows',
            'Duplicate text pairs (premise+hypothesis)',
            'Duplicate premises (non-empty only)'
        ],
        'Count': [duplicates, text_pair_dupes, premise_dupes],
        'Percentage': [
            f'{(duplicates/len(df)*100):.2f}%',
            f'{(text_pair_dupes/len(df)*100):.2f}%',
            f'{(premise_dupes/len(df[df["premise"] != ""])*100):.2f}%' if len(df[df['premise'] != '']) > 0 else '0.00%'
        ]
    }

    return pd.DataFrame(dup_data)


def t1_compute_statistics(df):
    """Compute comprehensive dataset statistics."""
    overview_data = {
        'Total Examples': [len(df)],
        'Unique Documents': [df['doc_id'].nunique()],
        'Unique Keys': [df['key'].nunique()],
        'Avg Examples/Doc': [f"{len(df) / df['doc_id'].nunique():.2f}"],
    }

    if 'binary_label' in df.columns:
        overview_data['Contradiction Count'] = [int(df['binary_label'].sum())]
        overview_data['Contradiction Rate'] = [f"{(df['binary_label'].sum()/len(df)*100):.2f}%"]

    if 'premise_len' in df.columns and 'hypothesis_len' in df.columns:
        overview_data['Avg Premise Length'] = [f"{df['premise_len'].mean():.1f}"]
        overview_data['Avg Hypothesis Length'] = [f"{df['hypothesis_len'].mean():.1f}"]
        overview_data['Avg Total Length'] = [f"{df['total_len'].mean():.1f}"]

    overview_df = pd.DataFrame(overview_data).T.reset_index()
    overview_df.columns = ['Metric', 'Value']

    label_df = None
    if 'binary_label' in df.columns:
        binary_counts = df['binary_label'].value_counts()
        binary_pct = (df['binary_label'].value_counts(normalize=True) * 100)

        label_df = pd.DataFrame({
            'Class': ['Non-Contradiction', 'Contradiction'],
            'Label': [0, 1],
            'Count': [binary_counts[0], binary_counts[1]],
            'Percentage': [f"{binary_pct[0]:.2f}%", f"{binary_pct[1]:.2f}%"]
        })

    return overview_df, label_df


def t1_analyze_text_lengths(df):
    """Analyze text lengths in the dataset."""
    df['premise_len'] = df['premise'].apply(lambda x: len(x.split()) if x else 0)
    df['hypothesis_len'] = df['hypothesis'].apply(lambda x: len(x.split()) if x else 0)
    df['total_len'] = df['premise_len'] + df['hypothesis_len']
    return df



def t1_split_data(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Split data into train, validation, and test sets. Groups by doc_id to prevent data leakage."""
    unique_docs = df['doc_id'].unique()

    train_docs, temp_docs = train_test_split(
        unique_docs,
        train_size=train_size,
        random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)
    val_docs, test_docs = train_test_split(
        temp_docs,
        train_size=val_ratio,
        random_state=random_state
    )

    train_df = df[df['doc_id'].isin(train_docs)].reset_index(drop=True)
    val_df = df[df['doc_id'].isin(val_docs)].reset_index(drop=True)
    test_df = df[df['doc_id'].isin(test_docs)].reset_index(drop=True)

    return train_df, val_df, test_df



def t1_create_tfidf_features(train_df, val_df, test_df,
                             max_features=5000, ngram_range=(1, 2)):
    """Create TF-IDF features for text pairs."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        lowercase=True
    )

    X_train = vectorizer.fit_transform(train_df['text_pair'])
    X_val = vectorizer.transform(val_df['text_pair'])
    X_test = vectorizer.transform(test_df['text_pair'])

    return vectorizer, X_train, X_val, X_test


def t1_create_bow_features(train_df, val_df, test_df,
                           max_features=5000, ngram_range=(1, 1)):
    """Create Bag-of-Words features for text pairs."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        lowercase=True
    )

    X_train = vectorizer.fit_transform(train_df['text_pair'])
    X_val = vectorizer.transform(val_df['text_pair'])
    X_test = vectorizer.transform(test_df['text_pair'])

    return vectorizer, X_train, X_val, X_test



def t1_plot_label_distribution(df, title='Label Distribution'):
    """Plot label distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    label_counts = df['label'].value_counts()
    axes[0].bar(label_counts.index, label_counts.values, color=['#e74c3c', '#3498db', '#95a5a6'])
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Original Label Distribution')
    axes[0].tick_params(axis='x', rotation=45)

    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')

    if 'binary_label' in df.columns:
        binary_counts = df['binary_label'].value_counts().sort_index()
        labels = ['Non-Contradiction', 'Contradiction']
        colors = ['#3498db', '#e74c3c']

        axes[1].bar(labels, binary_counts.values, color=colors)
        axes[1].set_xlabel('Label')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Binary Label Distribution')

        for i, v in enumerate(binary_counts.values):
            pct = v / len(df) * 100
            axes[1].text(i, v + 20, f'{v}\n({pct:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return fig


def t1_plot_text_lengths(df, title='Text Length Distribution'):
    """Plot text length distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(df['premise_len'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Premise Length (words)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Premise Length Distribution')
    axes[0, 0].axvline(df['premise_len'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["premise_len"].mean():.1f}')
    axes[0, 0].legend()

    axes[0, 1].hist(df['hypothesis_len'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Hypothesis Length (words)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Hypothesis Length Distribution')
    axes[0, 1].axvline(df['hypothesis_len'].mean(), color='blue', linestyle='--',
                       label=f'Mean: {df["hypothesis_len"].mean():.1f}')
    axes[0, 1].legend()

    axes[1, 0].hist(df['total_len'], bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Total Length (words)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Combined Text Length Distribution')
    axes[1, 0].axvline(df['total_len'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["total_len"].mean():.1f}')
    axes[1, 0].legend()

    scatter_colors = df['binary_label'].map({0: '#3498db', 1: '#e74c3c'}) if 'binary_label' in df.columns else '#95a5a6'
    axes[1, 1].scatter(df['premise_len'], df['hypothesis_len'],
                      c=scatter_colors, alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Premise Length (words)')
    axes[1, 1].set_ylabel('Hypothesis Length (words)')
    axes[1, 1].set_title('Premise vs Hypothesis Length')

    if 'binary_label' in df.columns:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Non-Contradiction'),
            Patch(facecolor='#e74c3c', label='Contradiction')
        ]
        axes[1, 1].legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

    return fig


def t1_plot_length_by_label(df, title='Text Length by Label'):
    """Plot text length comparison across labels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    text_types = [
        ('premise_len', 'Premise Length'),
        ('hypothesis_len', 'Hypothesis Length'),
        ('total_len', 'Total Length')
    ]

    for idx, (col, label) in enumerate(text_types):
        if 'binary_label' in df.columns:
            data = [
                df[df['binary_label'] == 0][col],
                df[df['binary_label'] == 1][col]
            ]
            axes[idx].boxplot(data, tick_labels=['Non-Contradiction', 'Contradiction'])
        else:
            labels_unique = df['label'].unique()
            data = [df[df['label'] == lbl][col] for lbl in labels_unique]
            axes[idx].boxplot(data, tick_labels=labels_unique)

        axes[idx].set_ylabel('Length (words)')
        axes[idx].set_title(label)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    return fig


def t1_get_examples(df, n_examples=5, label_filter=None):
    """Get example contradictions from the dataset."""
    if label_filter is not None:
        if 'binary_label' in df.columns:
            sample_df = df[df['binary_label'] == label_filter].sample(
                n=min(n_examples, len(df[df['binary_label'] == label_filter])),
                random_state=42
            )
        else:
            sample_df = df[df['label'] == label_filter].sample(
                n=min(n_examples, len(df[df['label'] == label_filter])),
                random_state=42
            )
    else:
        sample_df = df.sample(n=min(n_examples, len(df)), random_state=42)

    examples_data = []
    for idx, row in sample_df.iterrows():
        example = {
            'Document ID': row['doc_id'],
            'Key': row['key'],
            'Label': row['label'],
            'Premise': row['premise'][:200] + '...' if len(row['premise']) > 200 else row['premise'],
            'Hypothesis': row['hypothesis'][:200] + '...' if len(row['hypothesis']) > 200 else row['hypothesis']
        }
        if 'binary_label' in row:
            example['Binary Label'] = 'Contradiction' if row['binary_label'] == 1 else 'Non-Contradiction'
        examples_data.append(example)

    return pd.DataFrame(examples_data)


def t1_create_text_length_stats(df):
    """Create text length statistics summary."""
    text_stats = []

    for text_type in ['premise_len', 'hypothesis_len', 'total_len']:
        col_name = text_type.replace('_len', '').capitalize()
        stats_row = {
            'Text Type': col_name,
            'Min': int(df[text_type].min()),
            'Max': int(df[text_type].max()),
            'Mean': f"{df[text_type].mean():.1f}",
            'Median': f"{df[text_type].median():.1f}",
            'Std Dev': f"{df[text_type].std():.1f}"
        }
        text_stats.append(stats_row)

    return pd.DataFrame(text_stats)



def t1_run_full_pipeline(dataset_choice='english'):
    """Execute complete Task 1 pipeline."""
    if dataset_choice == 'english':
        train_raw, test_raw = load_english_dataset()
        df = train_raw.copy()
    else:
        raise NotImplementedError("Slovene dataset processing not yet implemented")

    df = t1_transform_to_binary(df)
    df = t1_combine_text_pairs(df)
    df = t1_analyze_text_lengths(df)

    quality_df = t1_check_missing_values(df)
    duplicate_df = t1_check_duplicates(df)
    overview_df, label_df = t1_compute_statistics(df)

    display(HTML("<h3>Data Quality Check</h3>"))
    display(quality_df)

    display(HTML("<h3>Duplicate Analysis</h3>"))
    display(duplicate_df)

    display(HTML("<h2>Dataset Overview</h2>"))
    display(overview_df)

    if label_df is not None:
        display(HTML("<h3>Binary Classification Distribution</h3>"))
        display(label_df)

    train_df, val_df, test_df = t1_split_data(df)

    tfidf_vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = t1_create_tfidf_features(
        train_df, val_df, test_df
    )

    bow_vectorizer, X_train_bow, X_val_bow, X_test_bow = t1_create_bow_features(
        train_df, val_df, test_df
    )

    fig_labels = t1_plot_label_distribution(df)
    fig_lengths = t1_plot_text_lengths(df)
    fig_length_by_label = t1_plot_length_by_label(df)

    examples_df = t1_get_examples(df, n_examples=5, label_filter=1)
    display(HTML("<h3>Example Contradictions</h3>"))
    display(examples_df)

    text_stats_df = t1_create_text_length_stats(df)
    display(HTML("<h3>Text Length Statistics</h3>"))
    display(text_stats_df)

    results = {
        'original_df': df,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'tfidf_vectorizer': tfidf_vectorizer,
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'bow_vectorizer': bow_vectorizer,
        'X_train_bow': X_train_bow,
        'X_val_bow': X_val_bow,
        'X_test_bow': X_test_bow,
        'y_train': train_df['binary_label'].values,
        'y_val': val_df['binary_label'].values,
        'y_test': test_df['binary_label'].values,
        'quality_df': quality_df,
        'duplicate_df': duplicate_df,
        'overview_df': overview_df,
        'label_df': label_df,
        'text_stats_df': text_stats_df
    }

    return results


# ============================================================================
# TASK 2: BASIC MACHINE LEARNING
# ============================================================================

class T2ModelTrainer:
    """Encapsulates training logic for different ML models."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def train_logistic_regression(self, X_train, y_train, **kwargs):
        """Train Logistic Regression model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = {
                'max_iter': kwargs.get('max_iter', 1000),
                'random_state': self.random_state,
                'class_weight': kwargs.get('class_weight', 'balanced'),
                'C': kwargs.get('C', 1.0),
                'penalty': kwargs.get('penalty', 'l2'),
                'solver': kwargs.get('solver', 'lbfgs')
            }
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', None),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
                'random_state': self.random_state,
                'class_weight': kwargs.get('class_weight', 'balanced'),
                'n_jobs': kwargs.get('n_jobs', -1),
                'verbose': 0
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
        return model

    def train_svm(self, X_train, y_train, **kwargs):
        """Train SVM model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel = kwargs.get('kernel', 'rbf')

            if kernel == 'linear' and X_train.shape[0] > 10000:
                params = {
                    'C': kwargs.get('C', 1.0),
                    'random_state': self.random_state,
                    'class_weight': kwargs.get('class_weight', 'balanced'),
                    'max_iter': kwargs.get('max_iter', 1000)
                }
                model = LinearSVC(**params)
            else:
                params = {
                    'C': kwargs.get('C', 1.0),
                    'kernel': kernel,
                    'gamma': kwargs.get('gamma', 'scale'),
                    'random_state': self.random_state,
                    'class_weight': kwargs.get('class_weight', 'balanced'),
                    'probability': True
                }
                model = SVC(**params)

            model.fit(X_train, y_train)
        return model

    def train_decision_tree(self, X_train, y_train, **kwargs):
        """Train Decision Tree model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = {
                'max_depth': kwargs.get('max_depth', None),
                'min_samples_split': kwargs.get('min_samples_split', 2),
                'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
                'random_state': self.random_state,
                'class_weight': kwargs.get('class_weight', 'balanced')
            }
            model = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train)
        return model


class T2HyperparameterTuner:
    """Handles hyperparameter tuning with GridSearchCV."""

    def __init__(self, cv=5, random_state=42, n_jobs=-1, verbose=0):
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def get_param_grid(self, model_type):
        """Get parameter grid for specific model type."""
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [1000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
            },
            'decision_tree': {
                'max_depth': [5, 10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': ['balanced', None]
            }
        }
        return param_grids.get(model_type, {})

    def tune_model(self, model, param_grid, X_train, y_train):
        """Perform grid search with cross-validation."""
        # Suppress all warnings during grid search
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
                scoring='f1',
                n_jobs=self.n_jobs,
                verbose=0,  # No console output
                return_train_score=True,
                error_score='raise'
            )

            grid_search.fit(X_train, y_train)

        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }


class T2ModelEvaluator:
    """Handles model evaluation and metric computation."""

    def evaluate(self, model, X, y, dataset_name='validation'):
        """Compute comprehensive evaluation metrics."""
        y_pred = model.predict(X)

        # Get probability predictions if available
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X)

        results = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred,
                                                           target_names=['Non-Contradiction', 'Contradiction'],
                                                           output_dict=True,
                                                           zero_division=0),
            'predictions': y_pred,
            'probabilities': y_prob
        }

        if y_prob is not None:
            try:
                results['roc_auc'] = roc_auc_score(y, y_prob)
                results['avg_precision'] = average_precision_score(y, y_prob)
            except:
                results['roc_auc'] = None
                results['avg_precision'] = None
        else:
            results['roc_auc'] = None
            results['avg_precision'] = None

        return results

    def create_confusion_matrix_df(self, cm, labels=['Non-Contradiction', 'Contradiction']):
        """Convert confusion matrix to formatted DataFrame."""
        cm_df = pd.DataFrame(
            cm,
            index=[f'True {label}' for label in labels],
            columns=[f'Pred {label}' for label in labels]
        )

        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        cm_display = pd.DataFrame(
            [[f'{cm[i,j]} ({cm_pct[i,j]:.1f}%)' for j in range(len(labels))]
             for i in range(len(labels))],
            index=[f'True {label}' for label in labels],
            columns=[f'Pred {label}' for label in labels]
        )

        return cm_display

    def create_classification_report_df(self, report_dict):
        """Convert classification report to DataFrame."""
        df = pd.DataFrame(report_dict).transpose()
        df = df.round(3)
        return df


class T2Visualizer:
    """Handles all Task 2 visualizations."""

    def __init__(self):
        self.colors = {
            'contradiction': '#e74c3c',
            'non_contradiction': '#3498db',
            'neutral': '#95a5a6',
            'success': '#2ecc71',
            'warning': '#f39c12'
        }
        self.model_colors = {
            'Logistic Regression': '#3498db',
            'Random Forest': '#2ecc71',
            'SVM': '#e74c3c',
            'Decision Tree': '#f39c12'
        }

    def plot_confusion_matrix(self, cm, model_name, normalize=True):
        """Plot confusion matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        if normalize:
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                       cbar_kws={'label': 'Percentage'},
                       xticklabels=['Non-Contradiction', 'Contradiction'],
                       yticklabels=['Non-Contradiction', 'Contradiction'],
                       ax=ax)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       cbar_kws={'label': 'Count'},
                       xticklabels=['Non-Contradiction', 'Contradiction'],
                       yticklabels=['Non-Contradiction', 'Contradiction'],
                       ax=ax)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()

        return fig

    def plot_confusion_matrices_grid(self, results_dict, feature_type='TF-IDF'):
        """Plot confusion matrices for all models in a grid."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'Decision Tree']

        for idx, (model_key, model_name) in enumerate(zip(
            ['logistic_regression', 'random_forest', 'svm', 'decision_tree'],
            model_names
        )):
            if model_key in results_dict:
                cm = results_dict[model_key]['confusion_matrix']
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                           xticklabels=['Non-Contr.', 'Contr.'],
                           yticklabels=['Non-Contr.', 'Contr.'],
                           ax=axes[idx], cbar_kws={'label': 'Percentage'})

                axes[idx].set_xlabel('Predicted Label')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_title(f'{model_name} ({feature_type})')

        plt.tight_layout()
        plt.show()
        return fig

    def plot_model_comparison(self, comparison_df, feature_type='TF-IDF'):
        """Plot model comparison across metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        models = comparison_df['model'].values
        x = np.arange(len(models))
        width = 0.6

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = comparison_df[metric].values
            colors = [self.model_colors.get(m, self.colors['neutral']) for m in models]

            bars = axes[idx].bar(x, values, width, color=colors, alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Model')
            axes[idx].set_ylabel(label)
            axes[idx].set_title(f'{label} Comparison ({feature_type})')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=45, ha='right')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.show()
        return fig

    def plot_roc_curves(self, models_dict, results_dict, X_test, y_test, feature_type='TF-IDF'):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_key, model_name in [
            ('logistic_regression', 'Logistic Regression'),
            ('random_forest', 'Random Forest'),
            ('svm', 'SVM'),
            ('decision_tree', 'Decision Tree')
        ]:
            if model_key in results_dict and results_dict[model_key]['probabilities'] is not None:
                y_prob = results_dict[model_key]['probabilities']
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
                       linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {feature_type} Features', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_precision_recall_curves(self, models_dict, results_dict, X_test, y_test, feature_type='TF-IDF'):
        """Plot Precision-Recall curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_key, model_name in [
            ('logistic_regression', 'Logistic Regression'),
            ('random_forest', 'Random Forest'),
            ('svm', 'SVM'),
            ('decision_tree', 'Decision Tree')
        ]:
            if model_key in results_dict and results_dict[model_key]['probabilities'] is not None:
                y_prob = results_dict[model_key]['probabilities']
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                avg_precision = average_precision_score(y_test, y_prob)

                ax.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})',
                       linewidth=2)

        baseline = y_test.sum() / len(y_test)
        ax.axhline(y=baseline, color='k', linestyle='--', linewidth=2,
                  label=f'Baseline (Random) = {baseline:.3f}')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves - {feature_type} Features', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.show()
        return fig

    def plot_feature_importance(self, model, vectorizer, model_name, top_n=20):
        """Plot top important features."""
        fig, ax = plt.subplots(figsize=(12, 8))

        feature_names = vectorizer.get_feature_names_out()

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]

            ax.barh(range(top_n), importances[indices], color=self.colors['success'], alpha=0.7)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {top_n} Features - {model_name}')

        elif hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            top_positive_idx = np.argsort(coefficients)[-top_n//2:]
            top_negative_idx = np.argsort(coefficients)[:top_n//2]
            top_idx = np.concatenate([top_negative_idx, top_positive_idx])

            colors_list = ['#e74c3c' if coefficients[i] < 0 else '#2ecc71' for i in top_idx]

            ax.barh(range(top_n), coefficients[top_idx], color=colors_list, alpha=0.7)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
            ax.set_xlabel('Coefficient Value')
            ax.set_title(f'Top {top_n} Features - {model_name}')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        else:
            ax.text(0.5, 0.5, 'Feature importance not available for this model',
                   ha='center', va='center', fontsize=14)
            ax.set_title(f'Feature Importance - {model_name}')

        plt.tight_layout()
        plt.show()
        return fig

    def plot_metrics_comparison_table(self, tfidf_comparison, bow_comparison):
        """Plot side-by-side comparison of TF-IDF vs BoW."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        for idx, (comparison_df, feature_type) in enumerate([
            (tfidf_comparison, 'TF-IDF'),
            (bow_comparison, 'BoW')
        ]):
            models = comparison_df['model'].values
            metrics = ['accuracy', 'precision', 'recall', 'f1']

            x = np.arange(len(models))
            width = 0.2

            for i, metric in enumerate(metrics):
                values = comparison_df[metric].values
                offset = width * (i - 1.5)
                axes[idx].bar(x + offset, values, width,
                            label=metric.capitalize(), alpha=0.8)

            axes[idx].set_xlabel('Model', fontsize=12)
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_title(f'{feature_type} Features - All Metrics', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(models, rotation=45, ha='right')
            axes[idx].legend(loc='lower right')
            axes[idx].set_ylim(0, 1.05)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig


class T2ResultsAggregator:
    """Aggregates and compares results from multiple models."""

    def compare_models(self, results_dict):
        """Create comparison DataFrame from evaluation results."""
        comparison_data = []

        for model_key, results in results_dict.items():
            model_name = model_key.replace('_', ' ').title()
            comparison_data.append({
                'model': model_name,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'roc_auc': results.get('roc_auc', None)
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)

        return comparison_df

    def identify_best_models(self, comparison_df):
        """Identify best model for each metric."""
        best_models = {}

        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in comparison_df.columns:
                best_idx = comparison_df[metric].idxmax()
                best_models[metric] = {
                    'model': comparison_df.loc[best_idx, 'model'],
                    'value': comparison_df.loc[best_idx, metric]
                }

        return best_models

    def create_summary_table(self, tfidf_comparison, bow_comparison):
        """Create comprehensive summary comparing TF-IDF and BoW."""
        tfidf_comparison = tfidf_comparison.copy()
        bow_comparison = bow_comparison.copy()

        tfidf_comparison['feature_type'] = 'TF-IDF'
        bow_comparison['feature_type'] = 'BoW'

        summary = pd.concat([tfidf_comparison, bow_comparison], ignore_index=True)
        summary = summary[['model', 'feature_type', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]

        return summary

    def aggregate_cv_results(self, tuning_results):
        """Create summary of cross-validation results."""
        cv_summary = []

        for model_name, results in tuning_results.items():
            model_display = model_name.replace('_', ' ').title()
            cv_summary.append({
                'Model': model_display,
                'Best CV F1': results['best_score'],
                'Best Parameters': str(results['best_params'])
            })

        cv_df = pd.DataFrame(cv_summary)
        cv_df = cv_df.round(4)

        return cv_df


def t2_run_full_pipeline(task1_results, tune_hyperparams=True, cv_folds=5, random_state=42):
    """
    Execute complete Task 2 pipeline for traditional ML classifiers.

    Args:
        task1_results: Dictionary from t1_run_full_pipeline()
        tune_hyperparams: Whether to perform hyperparameter tuning
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing trained models, evaluation results, and visualizations
    """

    display(HTML("<h2>Task 2: Traditional Machine Learning Models</h2>"))

    # Initialize components
    trainer = T2ModelTrainer(random_state=random_state)
    tuner = T2HyperparameterTuner(cv=cv_folds, random_state=random_state)
    evaluator = T2ModelEvaluator()
    visualizer = T2Visualizer()
    aggregator = T2ResultsAggregator()

    # Extract data from Task 1 results
    X_train_tfidf = task1_results['X_train_tfidf']
    X_val_tfidf = task1_results['X_val_tfidf']
    X_test_tfidf = task1_results['X_test_tfidf']

    X_train_bow = task1_results['X_train_bow']
    X_val_bow = task1_results['X_val_bow']
    X_test_bow = task1_results['X_test_bow']

    y_train = task1_results['y_train']
    y_val = task1_results['y_val']
    y_test = task1_results['y_test']

    tfidf_vectorizer = task1_results['tfidf_vectorizer']
    bow_vectorizer = task1_results['bow_vectorizer']

    results = {
        'models_tfidf': {},
        'models_bow': {},
        'val_results_tfidf': {},
        'val_results_bow': {},
        'test_results_tfidf': {},
        'test_results_bow': {},
        'tuning_results_tfidf': {},
        'tuning_results_bow': {}
    }

    # ========================================================================
    # PHASE 1: TF-IDF FEATURES
    # ========================================================================

    display(HTML("<h3>Part 1: TF-IDF Features</h3>"))

    # Train baseline models
    models_tfidf = {}
    models_tfidf['logistic_regression'] = trainer.train_logistic_regression(X_train_tfidf, y_train)
    models_tfidf['random_forest'] = trainer.train_random_forest(X_train_tfidf, y_train)
    models_tfidf['svm'] = trainer.train_svm(X_train_tfidf, y_train)
    models_tfidf['decision_tree'] = trainer.train_decision_tree(X_train_tfidf, y_train)

    # Hyperparameter tuning
    if tune_hyperparams:
        display(HTML("<h4>Hyperparameter Tuning Progress</h4>"))

        tuning_progress = []
        tuning_results_tfidf = {}

        for model_type in ['logistic_regression', 'random_forest', 'svm', 'decision_tree']:

            if model_type == 'logistic_regression':
                base_model = LogisticRegression(random_state=random_state)
            elif model_type == 'random_forest':
                base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            elif model_type == 'svm':
                base_model = SVC(probability=True, random_state=random_state)
            else:
                base_model = DecisionTreeClassifier(random_state=random_state)

            param_grid = tuner.get_param_grid(model_type)
            tuning_result = tuner.tune_model(base_model, param_grid, X_train_tfidf, y_train)

            tuning_results_tfidf[model_type] = tuning_result
            models_tfidf[model_type] = tuning_result['best_model']

            tuning_progress.append({
                'Model': model_type.replace('_', ' ').title(),
                'Best CV F1': f"{tuning_result['best_score']:.4f}",
                'Parameters Tested': len(tuning_result['cv_results'])
            })

        display(pd.DataFrame(tuning_progress))
        results['tuning_results_tfidf'] = tuning_results_tfidf

    # Evaluate on validation set
    val_results_tfidf = {}
    for model_key, model in models_tfidf.items():
        val_results_tfidf[model_key] = evaluator.evaluate(model, X_val_tfidf, y_val, 'validation')

    results['models_tfidf'] = models_tfidf
    results['val_results_tfidf'] = val_results_tfidf

    # Display validation results
    comparison_tfidf_val = aggregator.compare_models(val_results_tfidf)
    display(HTML("<h4>Validation Set Performance (TF-IDF)</h4>"))
    display(comparison_tfidf_val)

    # ========================================================================
    # PHASE 2: BAG-OF-WORDS FEATURES
    # ========================================================================

    display(HTML("<h3>Part 2: Bag-of-Words Features</h3>"))

    # Train baseline models
    models_bow = {}
    models_bow['logistic_regression'] = trainer.train_logistic_regression(X_train_bow, y_train)
    models_bow['random_forest'] = trainer.train_random_forest(X_train_bow, y_train)
    models_bow['svm'] = trainer.train_svm(X_train_bow, y_train)
    models_bow['decision_tree'] = trainer.train_decision_tree(X_train_bow, y_train)

    # Hyperparameter tuning
    if tune_hyperparams:
        tuning_progress = []
        tuning_results_bow = {}

        for model_type in ['logistic_regression', 'random_forest', 'svm', 'decision_tree']:

            if model_type == 'logistic_regression':
                base_model = LogisticRegression(random_state=random_state)
            elif model_type == 'random_forest':
                base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
            elif model_type == 'svm':
                base_model = SVC(probability=True, random_state=random_state)
            else:
                base_model = DecisionTreeClassifier(random_state=random_state)

            param_grid = tuner.get_param_grid(model_type)
            tuning_result = tuner.tune_model(base_model, param_grid, X_train_bow, y_train)

            tuning_results_bow[model_type] = tuning_result
            models_bow[model_type] = tuning_result['best_model']

            tuning_progress.append({
                'Model': model_type.replace('_', ' ').title(),
                'Best CV F1': f"{tuning_result['best_score']:.4f}",
                'Parameters Tested': len(tuning_result['cv_results'])
            })

        display(pd.DataFrame(tuning_progress))
        results['tuning_results_bow'] = tuning_results_bow

    # Evaluate on validation set
    val_results_bow = {}
    for model_key, model in models_bow.items():
        val_results_bow[model_key] = evaluator.evaluate(model, X_val_bow, y_val, 'validation')

    results['models_bow'] = models_bow
    results['val_results_bow'] = val_results_bow

    # Display validation results
    comparison_bow_val = aggregator.compare_models(val_results_bow)
    display(HTML("<h4>Validation Set Performance (BoW)</h4>"))
    display(comparison_bow_val)

    # ========================================================================
    # PHASE 3: TEST SET EVALUATION (FINAL)
    # ========================================================================

    display(HTML("<h3>Final Test Set Evaluation</h3>"))

    # Evaluate models on test set
    test_results_tfidf = {}
    for model_key, model in models_tfidf.items():
        test_results_tfidf[model_key] = evaluator.evaluate(model, X_test_tfidf, y_test, 'test')

    test_results_bow = {}
    for model_key, model in models_bow.items():
        test_results_bow[model_key] = evaluator.evaluate(model, X_test_bow, y_test, 'test')

    results['test_results_tfidf'] = test_results_tfidf
    results['test_results_bow'] = test_results_bow

    # Create comparison tables
    comparison_tfidf_test = aggregator.compare_models(test_results_tfidf)
    comparison_bow_test = aggregator.compare_models(test_results_bow)

    display(HTML("<h4>Test Set Performance (TF-IDF)</h4>"))
    display(comparison_tfidf_test)

    display(HTML("<h4>Test Set Performance (BoW)</h4>"))
    display(comparison_bow_test)

    # Overall summary
    summary_table = aggregator.create_summary_table(comparison_tfidf_test, comparison_bow_test)
    display(HTML("<h4>Overall Performance Summary</h4>"))
    display(summary_table)

    # Identify best models
    best_tfidf = aggregator.identify_best_models(comparison_tfidf_test)
    best_bow = aggregator.identify_best_models(comparison_bow_test)

    display(HTML("<h4>Best Models by Metric</h4>"))
    best_models_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        best_models_data.append({
            'Metric': metric.upper(),
            'Best TF-IDF Model': f"{best_tfidf[metric]['model']} ({best_tfidf[metric]['value']:.4f})",
            'Best BoW Model': f"{best_bow[metric]['model']} ({best_bow[metric]['value']:.4f})"
        })
    display(pd.DataFrame(best_models_data))

    # ========================================================================
    # PHASE 4: VISUALIZATIONS
    # ========================================================================

    display(HTML("<h3>Model Visualizations</h3>"))

    # Confusion matrices
    display(HTML("<h4>Confusion Matrices</h4>"))
    fig_cm_tfidf = visualizer.plot_confusion_matrices_grid(test_results_tfidf, 'TF-IDF')
    fig_cm_bow = visualizer.plot_confusion_matrices_grid(test_results_bow, 'BoW')

    # Model comparison
    display(HTML("<h4>Performance Metrics Comparison</h4>"))
    fig_comp_tfidf = visualizer.plot_model_comparison(comparison_tfidf_test, 'TF-IDF')
    fig_comp_bow = visualizer.plot_model_comparison(comparison_bow_test, 'BoW')

    # Side-by-side metrics comparison
    fig_metrics_table = visualizer.plot_metrics_comparison_table(comparison_tfidf_test, comparison_bow_test)

    # ROC curves
    display(HTML("<h4>ROC Curves</h4>"))
    fig_roc_tfidf = visualizer.plot_roc_curves(models_tfidf, test_results_tfidf, X_test_tfidf, y_test, 'TF-IDF')
    fig_roc_bow = visualizer.plot_roc_curves(models_bow, test_results_bow, X_test_bow, y_test, 'BoW')

    # Precision-Recall curves
    display(HTML("<h4>Precision-Recall Curves</h4>"))
    fig_pr_tfidf = visualizer.plot_precision_recall_curves(models_tfidf, test_results_tfidf, X_test_tfidf, y_test, 'TF-IDF')
    fig_pr_bow = visualizer.plot_precision_recall_curves(models_bow, test_results_bow, X_test_bow, y_test, 'BoW')

    # Feature importance
    display(HTML("<h4>Feature Importance</h4>"))

    # For Logistic Regression (TF-IDF)
    fig_imp_lr_tfidf = visualizer.plot_feature_importance(
        models_tfidf['logistic_regression'],
        tfidf_vectorizer,
        'Logistic Regression (TF-IDF)',
        top_n=20
    )

    # For Random Forest (TF-IDF)
    fig_imp_rf_tfidf = visualizer.plot_feature_importance(
        models_tfidf['random_forest'],
        tfidf_vectorizer,
        'Random Forest (TF-IDF)',
        top_n=20
    )

    # ========================================================================
    # DETAILED CLASSIFICATION REPORTS
    # ========================================================================
    display(HTML("<h3>Detailed Classification Reports</h3>"))

    # TF-IDF Reports
    display(HTML("<h4>TF-IDF Models - Test Set</h4>"))
    for model_key in ['logistic_regression', 'random_forest', 'svm', 'decision_tree']:
        model_name = model_key.replace('_', ' ').title()
        display(HTML(f"<h5>{model_name}</h5>"))

        report_df = evaluator.create_classification_report_df(
            test_results_tfidf[model_key]['classification_report']
        )
        display(report_df)

        cm_df = evaluator.create_confusion_matrix_df(
            test_results_tfidf[model_key]['confusion_matrix']
        )
        display(HTML("<p><b>Confusion Matrix:</b></p>"))
        display(cm_df)

    # BoW Reports
    display(HTML("<h4>Bag-of-Words Models - Test Set</h4>"))
    for model_key in ['logistic_regression', 'random_forest', 'svm', 'decision_tree']:
        model_name = model_key.replace('_', ' ').title()
        display(HTML(f"<h5>{model_name}</h5>"))

        report_df = evaluator.create_classification_report_df(
            test_results_bow[model_key]['classification_report']
        )
        display(report_df)

        cm_df = evaluator.create_confusion_matrix_df(
            test_results_bow[model_key]['confusion_matrix']
        )
        display(HTML("<p><b>Confusion Matrix:</b></p>"))
        display(cm_df)

    # ========================================================================
    # HYPERPARAMETER TUNING SUMMARY
    # ========================================================================
    if tune_hyperparams:
        display(HTML("<h3>Hyperparameter Tuning Summary</h3>"))

        cv_summary_tfidf = aggregator.aggregate_cv_results(tuning_results_tfidf)
        cv_summary_bow = aggregator.aggregate_cv_results(tuning_results_bow)

        display(HTML("<h4>TF-IDF - Best Hyperparameters</h4>"))
        display(cv_summary_tfidf)

        display(HTML("<h4>BoW - Best Hyperparameters</h4>"))
        display(cv_summary_bow)

    # Store all results
    results.update({
        'comparison_tfidf_val': comparison_tfidf_val,
        'comparison_bow_val': comparison_bow_val,
        'comparison_tfidf_test': comparison_tfidf_test,
        'comparison_bow_test': comparison_bow_test,
        'summary_table': summary_table,
        'best_models_tfidf': best_tfidf,
        'best_models_bow': best_bow
    })

    # Final summary with best overall model
    all_f1_scores = []
    all_f1_scores.extend([(f"{m} (TF-IDF)", comparison_tfidf_test[comparison_tfidf_test['model']==m]['f1'].values[0])
                          for m in comparison_tfidf_test['model']])
    all_f1_scores.extend([(f"{m} (BoW)", comparison_bow_test[comparison_bow_test['model']==m]['f1'].values[0])
                          for m in comparison_bow_test['model']])

    best_overall = max(all_f1_scores, key=lambda x: x[1])

    final_summary = pd.DataFrame([{
        'Best Overall Model': best_overall[0],
        'F1 Score': f"{best_overall[1]:.4f}",
        'Feature Type': 'TF-IDF' if 'TF-IDF' in best_overall[0] else 'BoW'
    }])

    display(HTML("<h3>🏆 Task 2 Summary</h3>"))
    display(final_summary)

    return results


if __name__ == "__main__":
    # Run Task 1
    task1_results = t1_run_full_pipeline(dataset_choice='english')

    # Run Task 2
    task2_results = t2_run_full_pipeline(
        task1_results=task1_results,
        tune_hyperparams=True,
        cv_folds=5,
        random_state=42
    )
