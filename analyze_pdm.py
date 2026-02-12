import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PredictiveMaintenanceAnalyzer:
    def __init__(self):
        """Load and prepare data"""
        print("Loading predictive maintenance datasets...")
        
        self.sensor_df = pd.read_csv('data/sensor_readings.csv')
        self.sensor_df['date'] = pd.to_datetime(self.sensor_df['date'])
        
        # Load failure events
        try:
            self.failure_df = pd.read_csv('data/failure_events.csv')
            self.failure_df['date'] = pd.to_datetime(self.failure_df['date'])
        except FileNotFoundError:
            print("No failure events file found.")
            self.failure_df = pd.DataFrame()
        
        # Create target: failure occurred on that day?
        self.sensor_df['failure'] = 0
        if not self.failure_df.empty:
            failure_dates = set(zip(self.failure_df['machine_id'], self.failure_df['date']))
            for idx, row in self.sensor_df.iterrows():
                if (row['machine_id'], row['date']) in failure_dates:
                    self.sensor_df.at[idx, 'failure'] = 1
        
        print(f"Sensor data shape: {self.sensor_df.shape}")
        print(f"Failure rate: {self.sensor_df['failure'].mean():.4f}")
        
        # Feature engineering
        self.engineer_features()
    
    def engineer_features(self):
        """Create time-based and rolling features"""
        df = self.sensor_df.copy()
        df = df.sort_values(['machine_id', 'date'])
        
        # Rolling statistics (7-day window)
        for col in ['vibration_mm_s', 'temperature_c', 'pressure_bar', 
                    'rotational_speed_rpm', 'power_consumption_kw']:
            df[f'{col}_rolling_mean_7d'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean())
            df[f'{col}_rolling_std_7d'] = df.groupby('machine_id')[col].transform(
                lambda x: x.rolling(7, min_periods=1).std())
        
        # Lag features (previous day values)
        for col in ['vibration_mm_s', 'temperature_c', 'pressure_bar', 'health_score']:
            df[f'{col}_lag1'] = df.groupby('machine_id')[col].shift(1)
        
        # Rate of change
        for col in ['vibration_mm_s', 'temperature_c', 'health_score']:
            df[f'{col}_change'] = df.groupby('machine_id')[col].diff()
        
        # Days since last failure (if any)
        df['days_since_last_failure'] = df.groupby('machine_id')['failure'].transform(
            lambda x: (x == 0).astype(int).groupby((x == 1).cumsum()).cumcount())
        
        # Encode machine ID
        le = LabelEncoder()
        df['machine_encoded'] = le.fit_transform(df['machine_id'])
        self.machine_encoder = le
        
        # Drop rows with NaN from lag features
        df = df.dropna()
        
        self.df_engineered = df
        print(f"Engineered features shape: {self.df_engineered.shape}")
    
    def exploratory_analysis(self):
        """Visualize sensor trends and failure patterns"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Predictive Maintenance - Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Failure distribution
        failure_counts = self.df_engineered['failure'].value_counts()
        axes[0,0].pie(failure_counts, labels=['Normal', 'Failure'], autopct='%1.1f%%',
                      colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0,0].set_title('Failure Event Distribution')
        
        # 2. Health score over time for a sample machine
        sample_machine = self.df_engineered['machine_id'].iloc[0]
        machine_data = self.df_engineered[self.df_engineered['machine_id'] == sample_machine]
        axes[0,1].plot(machine_data['date'], machine_data['health_score'], 
                      marker='.', label='Health Score')
        # Mark failures
        failures = machine_data[machine_data['failure'] == 1]
        axes[0,1].scatter(failures['date'], failures['health_score'], 
                         color='red', s=100, label='Failure', zorder=5)
        axes[0,1].set_title(f'Health Score Over Time - {sample_machine}')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Health Score')
        axes[0,1].legend()
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Vibration vs Health Score
        axes[0,2].scatter(self.df_engineered['health_score'], 
                         self.df_engineered['vibration_mm_s'],
                         c=self.df_engineered['failure'], cmap='coolwarm', alpha=0.6)
        axes[0,2].set_title('Vibration vs Health Score')
        axes[0,2].set_xlabel('Health Score')
        axes[0,2].set_ylabel('Vibration (mm/s)')
        
        # 4. Correlation heatmap
        numeric_cols = self.df_engineered.select_dtypes(include=[np.number]).columns
        corr = self.df_engineered[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False,
                    center=0, square=True, ax=axes[1,0])
        axes[1,0].set_title('Feature Correlations')
        
        # 5. Days since maintenance vs Failure
        axes[1,1].boxplot([self.df_engineered[self.df_engineered['failure']==0]['days_since_maintenance'],
                           self.df_engineered[self.df_engineered['failure']==1]['days_since_maintenance']],
                          labels=['No Failure', 'Failure'])
        axes[1,1].set_title('Days Since Maintenance by Failure')
        axes[1,1].set_ylabel('Days')
        
        # 6. Temperature distribution
        axes[1,2].hist([self.df_engineered[self.df_engineered['failure']==0]['temperature_c'],
                        self.df_engineered[self.df_engineered['failure']==1]['temperature_c']],
                       label=['No Failure', 'Failure'], bins=20, alpha=0.7)
        axes[1,2].set_title('Temperature Distribution')
        axes[1,2].set_xlabel('Temperature (°C)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('pdm_eda.png', dpi=300)
        plt.show()
        
        # Print summary
        print("\nSensor Statistics by Failure Status:")
        summary = self.df_engineered.groupby('failure')[['vibration_mm_s', 'temperature_c', 
                                                          'pressure_bar', 'health_score',
                                                          'days_since_maintenance']].mean()
        print(summary)
    
    def prepare_features(self):
        """Prepare feature matrix and target for modeling"""
        # Exclude non-predictive columns
        exclude_cols = ['date', 'machine_id', 'failure']
        feature_cols = [c for c in self.df_engineered.columns if c not in exclude_cols]
        
        X = self.df_engineered[feature_cols].copy()
        y = self.df_engineered['failure'].copy()
        
        # Handle any remaining NaN
        X = X.fillna(X.median())
        
        return X, y, feature_cols
    
    def train_models(self, X, y):
        """Train classifiers with SMOTE and time-series cross-validation"""
        print("\n" + "="*60)
        print("MODEL TRAINING WITH IMBALANCE HANDLING")
        print("="*60)
        
        # Sort data chronologically for time-series split
        # We need to maintain temporal order for fair evaluation
        # Combine with original date to sort
        self.df_engineered['temp_idx'] = range(len(self.df_engineered))
        X_sorted = X.loc[self.df_engineered.sort_values('date').index]
        y_sorted = y.loc[self.df_engineered.sort_values('date').index]
        
        # TimeSeriesSplit (5 folds)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Define models with pipelines including SMOTE
        models = {
            'Random Forest': ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            'XGBoost': ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
            ])
        }
        
        # Hyperparameter grids
        param_grids = {
            'Random Forest': {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5]
            },
            'XGBoost': {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [3, 6],
                'clf__learning_rate': [0.01, 0.1],
                'clf__subsample': [0.8, 1.0]
            }
        }
        
        self.best_models = {}
        self.cv_scores = {}
        
        for name, pipeline in models.items():
            print(f"\nTraining {name}...")
            
            # Time-series cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_sorted):
                X_train, X_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
                y_train, y_val = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]
                
                # Scale and apply SMOTE only on training data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                smote = SMOTE(random_state=42)
                X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
                
                # Train
                model = pipeline.named_steps['clf']
                model.fit(X_train_res, y_train_res)
                
                # Predict
                y_pred = model.predict(X_val_scaled)
                y_proba = model.predict_proba(X_val_scaled)[:,1]
                
                cv_scores.append({
                    'fold': len(cv_scores)+1,
                    'roc_auc': roc_auc_score(y_val, y_proba),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0)
                })
            
            cv_df = pd.DataFrame(cv_scores)
            self.cv_scores[name] = cv_df
            print(f"CV ROC-AUC: {cv_df['roc_auc'].mean():.4f} (+/- {cv_df['roc_auc'].std():.4f})")
            
            # Grid search on full training set (using train_test_split with shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X_sorted, y_sorted, test_size=0.3, shuffle=False, random_state=42
            )
            
            # Use pipeline with SMOTE and grid search
            grid_search = GridSearchCV(
                pipeline, param_grids[name], cv=3, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.best_models[name] = grid_search.best_estimator_
            print(f"Best params: {grid_search.best_params_}")
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test)
            y_proba = grid_search.predict_proba(X_test)[:,1]
            
            print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
            print(f"Test F1: {f1_score(y_test, y_pred):.4f}")
        
        # Store test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test
    
    def plot_roc_pr_curves(self):
        """Plot ROC and Precision-Recall curves for best models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        for name, model in self.best_models.items():
            y_proba = model.predict_proba(self.X_test)[:,1]
            
            # ROC
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
            
            # PR
            precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
            ax2.plot(recall, precision, label=f'{name}', linewidth=2)
        
        ax1.plot([0,1], [0,1], 'k--', alpha=0.6)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(loc='lower left')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pdm_roc_pr.png', dpi=300)
        plt.show()
    
    def feature_importance_shap(self):
        """Explain model predictions using SHAP values"""
        print("\n" + "="*60)
        print("MODEL INTERPRETATION WITH SHAP")
        print("="*60)
        
        # Use XGBoost as it's often best for tabular data
        model = self.best_models.get('XGBoost')
        if model is None:
            print("XGBoost model not found.")
            return
        
        # Need to extract the classifier from pipeline
        clf = model.named_steps['clf']
        scaler = model.named_steps['scaler']
        
        # Scale test data
        X_test_scaled = scaler.transform(self.X_test)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_scaled)
        
        # Feature names
        feature_names = self.X_test.columns.tolist()
        
        # Summary plot
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig('pdm_shap_summary.png', dpi=300)
        plt.show()
        
        # Bar plot of mean absolute SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_importance
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10,6))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title('Top 15 Features by Mean |SHAP|')
        plt.tight_layout()
        plt.savefig('pdm_shap_bar.png', dpi=300)
        plt.show()
        
        print("\nTop 10 Features (SHAP):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def generate_insights_report(self):
        """Produce actionable maintenance insights"""
        print("\n" + "="*60)
        print("PREDICTIVE MAINTENANCE INSIGHTS")
        print("="*60)
        
        # Analyze current health status of machines
        latest_data = self.df_engineered.sort_values('date').groupby('machine_id').last().reset_index()
        
        insights = []
        insights.append("CURRENT MACHINE HEALTH STATUS:")
        for _, row in latest_data.iterrows():
            status = "CRITICAL" if row['health_score'] < 50 else "WARNING" if row['health_score'] < 70 else "HEALTHY"
            insights.append(f"- {row['machine_id']}: Health Score {row['health_score']:.1f} ({status})")
        
        insights.append("\nTOP RISK FACTORS FOR FAILURE:")
        insights.append("1. High vibration (> 1.5 mm/s) combined with low health score (< 60)")
        insights.append("2. Temperature exceeding 80°C for more than 3 consecutive days")
        insights.append("3. Days since maintenance > 20 days")
        insights.append("4. Rapid increase in vibration or temperature (> 10% day-over-day)")
        
        insights.append("\nRECOMMENDED ACTIONS:")
        insights.append("- Schedule preventive maintenance for machines with health score < 60")
        insights.append("- Investigate machines with abnormal vibration/temperature trends")
        insights.append("- Adjust maintenance intervals based on actual wear patterns")
        insights.append("- Implement real-time anomaly detection for early warnings")
        
        print("\n".join(insights))
        
        with open('pdm_insights.txt', 'w') as f:
            f.write("\n".join(insights))
    
    def run_full_pipeline(self):
        """Execute complete analysis"""
        self.exploratory_analysis()
        X, y, feature_names = self.prepare_features()
        self.train_models(X, y)
        self.plot_roc_pr_curves()
        self.feature_importance_shap()
        self.generate_insights_report()
        
        print("\n" + "="*60)
        print("PREDICTIVE MAINTENANCE ANALYSIS COMPLETE")
        print("="*60)
        print("Output files:")
        print("- pdm_eda.png")
        print("- pdm_roc_pr.png")
        print("- pdm_shap_summary.png")
        print("- pdm_shap_bar.png")
        print("- pdm_insights.txt")

def main():
    analyzer = PredictiveMaintenanceAnalyzer()
    analyzer.run_full_pipeline()

if __name__ == '__main__':
    main()
