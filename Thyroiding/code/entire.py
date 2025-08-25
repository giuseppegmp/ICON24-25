import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QComboBox, QPushButton, QVBoxLayout, QWidget, QMessageBox
import sys

# Classe per colori nel terminale
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

# Funzione per percentuali nei grafici a torta
def autopct(pct):
    return ('%.2f' % pct + "%") if pct > 1 else ''

# Funzione per grafici di confronto delle metriche
def plot_metrics_comparison(results):
    print(style.YELLOW + "Creazione grafici di confronto delle metriche..." + style.RESET)
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    
    # Prepara i dati per i grafici
    means = {metric: [results[model]['metrics'][metric][0] for model in models] for metric in metrics_names}
    stds = {metric: [results[model]['metrics'][metric][1] for model in models] for metric in metrics_names}
    
    # Crea un grafico per ogni metrica
    for metric in metrics_names:
        plt.figure(figsize=(8, 6))
        x = np.arange(len(models))
        plt.bar(x, means[metric], yerr=stds[metric], capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        plt.xticks(x, models)
        plt.ylabel(metric.capitalize())
        plt.title(f'Confronto {metric.capitalize()} (CV) tra i modelli')
        plt.ylim(0, 1)
        for i, v in enumerate(means[metric]):
            plt.text(i, v + 0.02, f'{v:.3f}\n±{stds[metric][i]:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'metrics_comparison_{metric}.png')
        plt.show()
    print(style.GREEN + "Grafici di confronto salvati come 'metrics_comparison_*.png'." + style.RESET)

# Passo 1: Pre-elaborazione
def optimization_data(file_path):
    start_time = time.time()
    print(style.BLUE + "Dataset caricato." + style.RESET)
    df = pd.read_csv(file_path)
    pd.set_option('display.max_columns', None)
    print("Colonne del dataset:", df.columns.tolist())
    print("Dataset Info:")
    print(df.info())
    print("\nValori mancanti:\n", df.isnull().sum())
    
    # Definisci feature numeriche e categoriche
    numeric_features = ['Age']
    categorical_features = ['Gender', 'Hx Radiothreapy', 'Adenopathy', 'Pathology', 'Focality', 
                           'Risk', 'T', 'N', 'M', 'Stage', 'Response']
    
    # Escludi 'Response' dalle feature categoriche
    categorical_features = [col for col in categorical_features if col != 'Response']
    
    # Filtra le feature esistenti
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]
    print(f"Feature numeriche filtrate: {numeric_features}")
    print(f"Feature categoriche filtrate: {categorical_features}")
    
    # Stampa valori unici per feature categoriche
    for col in categorical_features:
        print(f"Valori unici per {col}: {df[col].unique().tolist()}")
    
    # Imputazione valori mancanti
    print(style.YELLOW + "Imputazione valori mancanti..." + style.RESET)
    if numeric_features:
        num_imputer = SimpleImputer(strategy="mean")
        df[numeric_features] = num_imputer.fit_transform(df[numeric_features])
    if categorical_features:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
    
    print("\nValori mancanti dopo imputazione:\n", df.isnull().sum())
    
    # Identifica la colonna target
    target_column = None
    possible_targets = [col for col in df.columns if 'recur' in col.lower()]
    if possible_targets:
        target_column = possible_targets[0]
        print(style.YELLOW + f"Colonna target trovata: {target_column}" + style.RESET)
    else:
        print(style.RED + "Errore: Nessuna colonna target trovata con 'recur' nel nome." + style.RESET)
        print("Colonne disponibili:", df.columns.tolist())
        raise ValueError("Colonna target non trovata nel dataset.")
    
    print(style.YELLOW + f"Valori unici in {target_column}:" + style.RESET)
    print(df[target_column].unique())
    df[target_column] = df[target_column].str.lower().str.strip()
    df[target_column] = df[target_column].map({'no': 0, 'yes': 1})
    
    if df[target_column].isnull().any():
        print(style.RED + f"Errore: Alcuni valori in {target_column} non sono stati mappati. Valori unici:" + style.RESET)
        print(df[target_column].unique())
        raise ValueError("Mappatura del target fallita.")
    
    # Visualizza distribuzione delle classi
    print(style.YELLOW + "Controllo del bilanciamento delle classi" + style.RESET)
    labels = ["No Recurrence", "Recurrence"]
    ax = df[target_column].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Distribuzione dei casi di recidiva tiroidea")
    plt.legend(labels=labels, loc="best")
    plt.savefig("class_distribution.png")
    plt.show()
    
    # Gestione sicura di value_counts
    vc = df[target_column].value_counts()
    no_recur_count = vc.get(0, 0)
    recur_count = vc.get(1, 0)
    total_count = no_recur_count + recur_count
    
    print(style.GREEN + "No Recurrence: " + style.RESET, no_recur_count,
          f'(% {no_recur_count / total_count * 100:.2f})' if total_count > 0 else '(0%)')
    print(style.RED + "Recurrence: " + style.RESET, recur_count,
          f'(% {recur_count / total_count * 100:.2f})' if total_count > 0 else '(0%)')
    
    # Bilanciamento classi
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_minority_upsampled, df_majority]).reset_index(drop=True)
    
    print(style.YELLOW + "\nValori dopo oversampling:" + style.RESET)
    vc_balanced = df_balanced[target_column].value_counts()
    no_recur_count_balanced = vc_balanced.get(0, 0)
    recur_count_balanced = vc_balanced.get(1, 0)
    total_count_balanced = no_recur_count_balanced + recur_count_balanced
    
    print(style.GREEN + "No Recurrence: " + style.RESET, no_recur_count_balanced,
          f'(% {no_recur_count_balanced / total_count_balanced * 100:.2f})' if total_count_balanced > 0 else '(0%)')
    print(style.RED + "Recurrence: " + style.RESET, recur_count_balanced,
          f'(% {recur_count_balanced / total_count_balanced * 100:.2f})' if total_count_balanced > 0 else '(0%)')
    
    ax = df_balanced[target_column].value_counts().plot(kind='pie', figsize=(5, 5), autopct=autopct, labels=None)
    ax.axes.get_yaxis().set_visible(False)
    plt.title("Distribuzione dei casi dopo oversampling")
    plt.legend(labels=labels, loc="best")
    plt.savefig("class_distribution_oversampled.png")
    plt.show()
    
    print(f"Tempo pre-elaborazione: {time.time() - start_time:.2f} secondi")
    return df_balanced, numeric_features, categorical_features, target_column

# Passo 2: Caricamento e suddivisione
def load_dataset(df, numeric_features, categorical_features, target_column):
    start_time = time.time()
    print(style.YELLOW + "Caricamento e suddivisione dataset..." + style.RESET)
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    
    # Ripristino indice per garantire unicità
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    if categorical_features:
        encoded_cats = encoder.fit_transform(X[categorical_features])
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features), index=X.index)
        X = pd.concat([X[numeric_features], encoded_df], axis=1)
    
    # Imputazione dopo codifica
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    # Identifica feature costanti
    variances = X.var()
    constant_features = variances[variances == 0].index.tolist()
    if constant_features:
        print(f"Feature costanti rimosse: {constant_features}")
        X = X.drop(columns=constant_features)
    
    scaler = StandardScaler()
    if numeric_features:
        X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Suddivisione senza campionamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=25)
    
    # Selezione feature
    k = min(10, X_train.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"Feature selezionate: {selected_features}")
    
    # Verifica NaN
    print(f"NaN in X_train_selected: {np.any(np.isnan(X_train_selected))}")
    print(f"NaN in X_test_selected: {np.any(np.isnan(X_test_selected))}")
    
    print(f"Shape di X_train: {X_train_selected.shape}")
    print(f"Shape di X_test: {X_test_selected.shape}")
    
    print(f"Tempo caricamento dataset: {time.time() - start_time:.2f} secondi")
    return X_train_selected, X_test_selected, y_train, y_test, X, y, df, scaler, encoder, selected_features

# Passo 3: Apprendimento supervisionato

def train_supervised_models(X_train, y_train, X_test, y_test):
    start_time = time.time()
    print("\033[33mAddestramento modelli supervisionati...\033[0m")
    models = {
        "Random Forest": (RandomForestClassifier(random_state=42, class_weight="balanced"), {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }),
        "k-NN": (KNeighborsClassifier(), {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "p": [1, 2, 3],
            "leaf_size": [20, 30, 40]
        }),
        "XGBoost": (XGBClassifier(random_state=42), {
            "max_depth": [3, 6, 8, 10],
            "n_estimators": [100, 200, 300, 400],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0]
        }),
        "Decision Tree": (DecisionTreeClassifier(random_state=42, class_weight="balanced"), {
            "max_depth": [3, 5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["gini", "entropy"]
        })
    }
    
    results = {}
    best_model = None
    best_model_name = None
    best_f1_score = -1
    
    for name, (model, param_grid) in models.items():
        print(f"Addestramento {name}...")
        grid = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=30, cv=5, scoring="f1", n_jobs=1, random_state=42)
        grid.fit(X_train, y_train)
        
        # Calcola metriche con validazione incrociata
        metrics = {
            'accuracy': cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='accuracy'),
            'precision': cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='precision'),
            'recall': cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='recall'),
            'f1': cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='f1')
        }
        
        results[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "metrics": {
                'accuracy': (metrics['accuracy'].mean(), metrics['accuracy'].std()),
                'precision': (metrics['precision'].mean(), metrics['precision'].std()),
                'recall': (metrics['recall'].mean(), metrics['recall'].std()),
                'f1': (metrics['f1'].mean(), metrics['f1'].std())
            }
        }
        
        # Aggiorna il miglior modello
        current_f1 = metrics['f1'].mean()
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_model = grid.best_estimator_
            best_model_name = name
        
        print(f"{name} - Migliori parametri: {grid.best_params_}")
        print(f"{name} - Metriche (CV):")
        print(f"  Accuracy: {results[name]['metrics']['accuracy'][0]:.3f}")
        print(f"  Precision: {results[name]['metrics']['precision'][0]:.3f}")
        print(f"  Recall: {results[name]['metrics']['recall'][0]:.3f}")
        print(f"  F1-score: {results[name]['metrics']['f1'][0]:.3f}")
        
        print(f"{name} - Deviazione Standard (CV):")
        print(f"  Accuracy: ±{results[name]['metrics']['accuracy'][1]:.3f}")
        print(f"  Precision: ±{results[name]['metrics']['precision'][1]:.3f}")
        print(f"  Recall: ±{results[name]['metrics']['recall'][1]:.3f}")
        print(f"  F1-score: ±{results[name]['metrics']['f1'][1]:.3f}")
        
        y_pred = grid.best_estimator_.predict(X_test)
        print(f"\nClassification Report - {name} (Test):")
        report = classification_report(y_test, y_pred, target_names=["No Recurrence", "Recurrence"], output_dict=True, zero_division=0)
        print(f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        for label in ["No Recurrence", "Recurrence"]:
            print(f"{label:>20} {report[label]['precision']:>10.2f} {report[label]['recall']:>10.2f} {report[label]['f1-score']:>10.2f} {int(report[label]['support']):>10}")
        print(f"\n{'accuracy':>20} {report['accuracy']:>10.2f} {'':>10} {'':>10} {int(sum(report[label]['support'] for label in report if label not in ('accuracy', 'macro avg', 'weighted avg'))):>10}")
        
        #cm = confusion_matrix(y_test, y_pred)
        #sns.heatmap(cm, annot=True, fmt="d", xticklabels=["No Recurrence", "Recurrence"], yticklabels=["No Recurrence", "Recurrence"])
        #plt.title(f"Matrice di confusione - {name}")
        #plt.savefig(f"confusion_matrix_{name}.png")
        #plt.show()
        
        # Curva ROC
        if hasattr(grid.best_estimator_, "predict_proba"):
            y_pred_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend()
            plt.savefig(f"roc_curve_{name}.png")
            plt.show()
    
    print(f"\033[32mMiglior modello: {best_model_name} con F1-score CV: {best_f1_score:.3f}\033[0m")
    
    # Crea grafici di confronto
    plot_metrics_comparison(results)
    
    print(f"Tempo addestramento modelli: {time.time() - start_time:.2f} secondi")
    return results, best_model, best_model_name
def user_interaction_gui(scaler, encoder, best_model, best_model_name, numeric_features, categorical_features, X_columns, selected_features):
    valid_values = {
        'Gender': ['M', 'F'],
        'Hx Radiothreapy': ['Yes', 'No'],
        'Adenopathy': ['Yes', 'No'],
        'Pathology': ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'],
        'Focality': ['Uni-Focal', 'Multi-Focal'],
        'Risk': ['Low', 'Intermediate', 'High'],
        'T': ['T1', 'T2', 'T3', 'T4'],
        'N': ['N0', 'N1'],
        'M': ['M0', 'M1'],
        'Stage': ['I', 'II', 'III', 'IV']
    }

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Previsione Recidiva Cancro alla Tiroide")
    window.setGeometry(100, 100, 400, 600)

    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    layout = QVBoxLayout()
    central_widget.setLayout(layout)

    # Etichetta per indicare il modello usato
    layout.addWidget(QLabel(f"Modello utilizzato: {best_model_name}"))

    input_vars = {}
    for feature in numeric_features:
        layout.addWidget(QLabel(f"{feature} (es. 30 per Age):"))
        entry = QLineEdit()
        layout.addWidget(entry)
        input_vars[feature] = entry

    for feature in categorical_features:
        layout.addWidget(QLabel(f"{feature}:"))
        combo = QComboBox()
        combo.addItems(valid_values.get(feature, []))
        layout.addWidget(combo)
        input_vars[feature] = combo

    def make_prediction():
        input_data = {}
        try:
            for feature in numeric_features:
                value = input_vars[feature].text()
                if not value:
                    QMessageBox.critical(window, "Errore", f"Inserisci un valore per {feature}")
                    return
                try:
                    value = float(value)
                    if feature == 'Age' and (value < 0 or value > 120):
                        QMessageBox.critical(window, "Errore", "L'età deve essere compresa tra 0 e 120 anni")
                        return
                    input_data[feature] = value
                except ValueError:
                    QMessageBox.critical(window, "Errore", f"Inserisci un valore numerico valido per {feature}")
                    return

            for feature in categorical_features:
                value = input_vars[feature].currentText()
                if not value:
                    QMessageBox.critical(window, "Errore", f"Seleziona un valore per {feature}")
                    return
                input_data[feature] = value

            input_df = pd.DataFrame([input_data])
            if numeric_features:
                input_df[numeric_features] = scaler.transform(input_df[numeric_features])
            if categorical_features:
                try:
                    encoded_cats = encoder.transform(input_df[categorical_features])
                    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
                    input_processed = pd.concat([input_df[numeric_features], encoded_df], axis=1)
                except ValueError as e:
                    QMessageBox.critical(window, "Errore", f"Errore nella codifica: {e}")
                    return
            else:
                input_processed = input_df[numeric_features].copy()

            for col in X_columns:
                if col not in input_processed.columns:
                    input_processed[col] = 0
            input_processed = input_processed[X_columns]
            input_processed = input_processed[selected_features]
            input_processed_array = input_processed.to_numpy()

            prediction = best_model.predict(input_processed_array)
            probability = best_model.predict_proba(input_processed_array)[0] if hasattr(best_model, "predict_proba") else None

            result = "Recurrence" if prediction[0] == 1 else "No Recurrence"
            result_text = f"Predizione ({best_model_name}): {result}"
            if probability is not None:
                result_text += f"\nProbabilità: {probability.max():.2%}"
            result_label.setText(result_text)
            result_label.setStyleSheet("color: red" if prediction[0] == 1 else "color: green")
        except Exception as e:
            QMessageBox.critical(window, "Errore", f"Errore durante la predizione: {e}")

    predict_button = QPushButton("Fai Predizione")
    predict_button.clicked.connect(make_prediction)
    layout.addWidget(predict_button)

    result_label = QLabel("Inserisci i dati e clicca su 'Fai Predizione'")
    layout.addWidget(result_label)

    exit_button = QPushButton("Esci")
    exit_button.clicked.connect(app.quit)
    layout.addWidget(exit_button)

    window.show()
    sys.exit(app.exec_())

# Passo 5: Main
def main():
    start_time = time.time()
    print("\033[34mInizio esecuzione del progetto...\033[0m")
    #file_path = "/Users/giuseppegiampietro/Downloads/filtered_thyroid_data.csv"
    file_path = os.path.join(os.path.dirname(__file__), "filtered_thyroid_data.csv")
    df, numeric_features, categorical_features, target_column = optimization_data(file_path)
    X_train, X_test, y_train, y_test, X, y, df, scaler, encoder, selected_features = load_dataset(df, numeric_features, categorical_features, target_column)
    
    print(f"Shape di X_train: {X_train.shape}")
    print(f"Shape di X_test: {X_test.shape}")
    print(f"selected_features: {selected_features}")
    
    results, best_model, best_model_name = train_supervised_models(X_train, y_train, X_test, y_test)
    
    # Avvia la GUI con il miglior modello
    user_interaction_gui(scaler, encoder, best_model, best_model_name, numeric_features, 
                        categorical_features, X.columns, selected_features)
    
    print(f"Tempo totale esecuzione: {time.time() - start_time:.2f} secondi")

if __name__ == "__main__":
    main()