import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.inspection import permutation_importance

def create_sequential_features(df, sequence_length=5):
    """Create sequential features without data leakage"""
    features = []
    targets = []
    
    # Sort by date to maintain temporal order
    df = df.sort_values('Date').reset_index(drop=True)
    
    for i in range(sequence_length, len(df)):
        # Get sequence of previous games (only past data)
        sequence = df.iloc[i-sequence_length:i]
        current_game = df.iloc[i]
        
        # Only use past information for features
        recent_features = {
            'rolling_avg_point_diff': sequence['point_diff'].mean(),
            'rolling_avg_fg_diff': sequence['fg_pct_diff'].mean(),
            'rolling_std_point_diff': sequence['point_diff'].std(),
            'home_win_streak': sequence['HOME_TEAM_WINS'].sum(),
        }
        
        # Add rolling averages for other available features
        if 'ft_pct_diff' in df.columns:
            recent_features['rolling_avg_ft_diff'] = sequence['ft_pct_diff'].mean()
        if 'fg3_pct_diff' in df.columns:
            recent_features['rolling_avg_fg3_diff'] = sequence['fg3_pct_diff'].mean()
        if 'ast_diff' in df.columns:
            recent_features['rolling_avg_ast_diff'] = sequence['ast_diff'].mean()
        if 'reb_diff' in df.columns:
            recent_features['rolling_avg_reb_diff'] = sequence['reb_diff'].mean()
        if 'implied_prob_diff' in df.columns:
            recent_features['rolling_avg_prob_diff'] = sequence['implied_prob_diff'].mean()
        
        # Current game features (only pre-game information)
        current_features = {}
        
        # Only include features available BEFORE the game
        pre_game_features = ['implied_prob_diff', 'spread']  # These are known before game
        
        for feature in pre_game_features:
            if feature in current_game:
                current_features[feature] = current_game[feature]
        
        # Combine all features (only pre-game info)
        combined_features = {**current_features, **recent_features}
        features.append(list(combined_features.values()))
        targets.append(current_game['HOME_TEAM_WINS'])
    
    feature_names = list(combined_features.keys()) if features else []
    return np.array(features), np.array(targets), feature_names

def load_and_prepare_data(file_path):
    """Load and prepare the merged dataset for training"""
    print("📊 Loading merged dataset...")
    df = pd.read_csv(file_path)
    
    # Check if we have the required columns
    if 'HOME_TEAM_WINS' not in df.columns:
        print("❌ Target column 'HOME_TEAM_WINS' not found!")
        return None, []
    
    # Convert Date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove features that cause data leakage
    # These are game statistics that are only known AFTER the game
    leakage_features = ['point_diff', 'fg_pct_diff', 'ft_pct_diff', 'fg3_pct_diff', 
                       'ast_diff', 'reb_diff', 'PTS_home', 'PTS_away', 'FG_PCT_home', 
                       'FG_PCT_away', 'FT_PCT_home', 'FT_PCT_away', 'FG3_PCT_home', 
                       'FG3_PCT_away', 'AST_home', 'AST_away', 'REB_home', 'REB_away']
    
    # Only use features available BEFORE the game
    safe_features = ['implied_prob_diff', 'spread', 'total', 'home_win_prob', 'away_win_prob']
    
    # Only include features that actually exist in the dataframe and are safe
    available_features = []
    for feature in safe_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            print(f"⚠️ Feature '{feature}' not found in dataset")
    
    print(f"🔒 Using only pre-game features: {available_features}")
    
    # Remove rows with missing values
    df_clean = df[available_features + ['HOME_TEAM_WINS', 'Date']].dropna()
    
    print(f"✅ Data prepared: {len(df_clean)} samples, {len(available_features)} features")
    print(f"📅 Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    
    return df_clean, available_features

def temporal_train_test_split(df, test_size=0.2):
    """Split data temporally to avoid data leakage"""
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df.drop(['HOME_TEAM_WINS', 'Date'], axis=1)
    X_test = test_df.drop(['HOME_TEAM_WINS', 'Date'], axis=1)
    y_train = train_df['HOME_TEAM_WINS']
    y_test = test_df['HOME_TEAM_WINS']
    
    print(f"📅 Training period: {train_df['Date'].min()} to {train_df['Date'].max()}")
    print(f"📅 Test period: {test_df['Date'].min()} to {test_df['Date'].max()}")
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest classifier"""
    print("🌲 Training Random Forest...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Reduced to prevent overfitting
        max_depth=10,
        min_samples_split=20,  # Increased to prevent overfitting
        min_samples_leaf=10,   # Increased to prevent overfitting
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Random Forest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"📊 Confusion Matrix:\n{cm}")
    
    # Feature importance
    if hasattr(rf_model, 'feature_importances_'):
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Feature Importances:")
        print(feature_imp)
    else:
        feature_imp = None
    
    return rf_model, accuracy, feature_imp

def train_neural_network(X_train, X_test, y_train, y_test):
    """Train Neural Network classifier"""
    print("🧠 Training Neural Network...")
    
    # Scale the data for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    nn_model = MLPClassifier(
        hidden_layer_sizes=(50, 25),  # Reduced complexity
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        learning_rate='adaptive',
        alpha=0.01  # Regularization
    )
    
    nn_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = nn_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Neural Network Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"📊 Confusion Matrix:\n{cm}")
    
    return nn_model, scaler, accuracy

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression as a simple baseline"""
    print("📊 Training Logistic Regression (Baseline)...")
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=0.1,  # Regularization
        penalty='l2'
    )
    
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Logistic Regression Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"📊 Confusion Matrix:\n{cm}")
    
    return lr_model, scaler, accuracy

def calculate_baseline_accuracy(y_test):
    """Calculate baseline accuracy (always predict majority class)"""
    majority_class = y_test.mode()[0]
    baseline_pred = [majority_class] * len(y_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    print(f"📈 Baseline accuracy (always predict {majority_class}): {baseline_accuracy:.4f}")
    return baseline_accuracy

def plot_results(accuracies, baseline_accuracy, feature_importance=None):
    """Plot model results"""
    plt.figure(figsize=(15, 5))
    
    # Model comparison
    plt.subplot(1, 3, 1)
    models = list(accuracies.keys())
    scores = list(accuracies.values())
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = plt.bar(models, scores, color=colors[:len(models)])
    
    # Add baseline line
    plt.axhline(y=baseline_accuracy, color='red', linestyle='--', label=f'Baseline ({baseline_accuracy:.4f})')
    
    plt.ylabel('Accuracy')
    plt.title('Model Comparison vs Baseline')
    plt.ylim(0, 1)
    plt.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Feature importance
    if feature_importance is not None and not feature_importance.empty:
        plt.subplot(1, 3, 2)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.gca().invert_yaxis()
    
    # Accuracy distribution
    plt.subplot(1, 3, 3)
    acc_values = list(accuracies.values())
    plt.hist(acc_values, bins=10, alpha=0.7, color='lightblue')
    plt.axvline(baseline_accuracy, color='red', linestyle='--', label='Baseline')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../models/model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load and prepare data
    data_path = "../data/processed/merged_features.csv"
    df, feature_names = load_and_prepare_data(data_path)
    
    if df is None or len(df) == 0:
        print("❌ No data available for training")
        return
    
    # Temporal train-test split
    X_train, X_test, y_train, y_test = temporal_train_test_split(df)
    
    # Calculate baseline accuracy
    baseline_accuracy = calculate_baseline_accuracy(y_test)
    
    # Train models
    accuracies = {}
    models = {}
    feature_imp = None
    
    # Random Forest
    rf_model, rf_acc, feature_imp = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    accuracies['Random Forest'] = rf_acc
    models['rf'] = rf_model
    
    # Neural Network
    nn_model, nn_scaler, nn_acc = train_neural_network(X_train, X_test, y_train, y_test)
    accuracies['Neural Network'] = nn_acc
    models['nn'] = nn_model
    models['nn_scaler'] = nn_scaler
    
    # Logistic Regression (Baseline)
    lr_model, lr_scaler, lr_acc = train_logistic_regression(X_train, X_test, y_train, y_test)
    accuracies['Logistic Regression'] = lr_acc
    models['lr'] = lr_model
    models['lr_scaler'] = lr_scaler
    
    # Time Series Cross Validation
    print("\n🔍 Time Series Cross-Validation for Random Forest:")
    tscv = TimeSeriesSplit(n_splits=5)
    X_all = df.drop(['HOME_TEAM_WINS', 'Date'], axis=1)
    y_all = df['HOME_TEAM_WINS']
    
    cv_scores = cross_val_score(rf_model, X_all, y_all, cv=tscv, scoring='accuracy')
    print(f"Time Series CV Scores: {cv_scores}")
    print(f"Time Series CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save models
    os.makedirs('../models', exist_ok=True)
    joblib.dump(rf_model, '../models/random_forest.pkl')
    joblib.dump(nn_model, '../models/neural_network.pkl')
    joblib.dump(nn_scaler, '../models/nn_scaler.pkl')
    joblib.dump(lr_model, '../models/logistic_regression.pkl')
    joblib.dump(lr_scaler, '../models/lr_scaler.pkl')
    
    # Save feature names
    with open('../models/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    # Plot results
    plot_results(accuracies, baseline_accuracy, feature_imp)
    
    print("\n🎯 Final Model Accuracies:")
    for model, acc in accuracies.items():
        improvement = acc - baseline_accuracy
        print(f"  {model}: {acc:.4f} (vs baseline: {improvement:+.4f})")
    
    # Check for potential issues
    print("\n🔍 Data Quality Check:")
    print(f"  Target distribution: {y_test.value_counts().to_dict()}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")

if __name__ == "__main__":
    main()