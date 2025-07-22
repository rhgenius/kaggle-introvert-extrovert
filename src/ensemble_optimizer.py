import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class EnsembleOptimizer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.optimized_models = {}
        self.ensemble_model = None
        self.cv_scores = {}
        
    def create_base_models(self):
        """Create base models with default parameters"""
        self.base_models = {
            'rf': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'xgb': xgb.XGBClassifier(
                random_state=self.random_state, 
                eval_metric='logloss',
                enable_categorical=True,  # Aktifkan fitur kategorikal
                tree_method='hist',
                verbosity=0,
                use_label_encoder=False,
                objective='binary:logistic'
            ),
            'lgb': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            'cat': CatBoostClassifier(random_seed=self.random_state, verbose=False),
            'lr_none': LogisticRegression(random_state=self.random_state, max_iter=2000, penalty=None),
            'lr_l1': LogisticRegression(random_state=self.random_state, max_iter=2000, penalty='l1', solver='saga'),
            'lr_l2': LogisticRegression(random_state=self.random_state, max_iter=2000, penalty='l2'),
            'lr_elasticnet': LogisticRegression(random_state=self.random_state, max_iter=2000, penalty='elasticnet', solver='saga', l1_ratio=0.5)
        }
    
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters for each base model"""
        print("Optimizing hyperparameters...")
        
        self.optimized_models = {}
        self.cv_scores = {}
        
        # Gunakan StratifiedKFold dengan lebih banyak fold untuk validasi yang lebih stabil
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Get parameter grids
        param_grids = self.get_param_grids()
        
        # Mapping dari nama model di base_models ke nama di param_grids
        model_name_mapping = {
            'rf': 'random_forest',
            'lgb': 'lightgbm',
            'cat': 'catboost',
            'lr_none': 'logistic',
            'lr_l1': 'logistic_l1',
            'lr_l2': 'logistic_l2',
            'lr_elasticnet': 'logistic_elasticnet',
            'xgb': 'xgboost'
        }
        
        for model_name, model in self.base_models.items():
            print(f"\nOptimizing {model_name}...")
            
            # Gunakan mapping untuk mendapatkan nama yang sesuai di param_grids
            param_grid_name = model_name_mapping.get(model_name, model_name)
            
            # Pastikan nama model ada di param_grids
            if param_grid_name not in param_grids:
                print(f"Warning: No parameter grid found for {model_name}, skipping optimization")
                continue
                
            param_grid = param_grids[param_grid_name]
            
            # Gunakan RandomizedSearchCV dengan lebih banyak iterasi dan scoring yang lebih komprehensif
            search = RandomizedSearchCV(
                model, param_grid, n_iter=50,
                cv=cv, 
                scoring=['accuracy', 'f1', 'roc_auc'],
                refit='accuracy',  # Refit berdasarkan accuracy
                n_jobs=-1, 
                verbose=1, 
                random_state=self.random_state,
                return_train_score=True  # Tambahkan untuk memeriksa overfitting
            )
            
            search.fit(X_train, y_train)
            
            self.optimized_models[model_name] = search.best_estimator_
            self.cv_scores[model_name] = search.best_score_
            
            print(f"Best {model_name} CV score: {search.best_score_:.4f}")
            print(f"Best parameters: {search.best_params_}")
            
            # Tambahkan analisis overfitting
            train_score = search.cv_results_['mean_train_accuracy'][search.best_index_]
            test_score = search.best_score_
            print(f"Train score: {train_score:.4f}, Test score: {test_score:.4f}, Gap: {train_score - test_score:.4f}")
        
        return self.optimized_models, self.cv_scores
    
    def create_voting_ensemble(self, X_train, y_train, top_n=5):
        """Create voting ensemble from top models"""
        print(f"Creating voting ensemble with top {top_n} models...")
        
        # Sort models by CV score
        sorted_models = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:top_n]
        
        estimators = [(name, self.optimized_models[name]) for name, _ in top_models]
        
        # Tambahkan weights berdasarkan CV score untuk voting yang lebih baik
        weights = [score for _, score in top_models]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,  # Tambahkan weights
            n_jobs=-1
        )
        
        # Cross-validate ensemble dengan stratified K-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Voting ensemble CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return voting_clf, cv_scores.mean()
    
    def create_stacking_ensemble(self, X_train, y_train):
        """Create stacking ensemble"""
        print("Creating stacking ensemble...")
        
        # Gunakan meta-learner yang lebih kuat dengan parameter yang dioptimalkan
        meta_learner = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )
        
        # Gunakan semua model yang dioptimalkan
        estimators = [(name, model) for name, model in self.optimized_models.items()]
        
        # Buat stacking classifier dengan cross-validation yang lebih baik
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # Tingkatkan jumlah fold
            stack_method='auto',
            n_jobs=-1,
            passthrough=True  # Tambahkan fitur asli ke meta-learner
        )
        
        # Evaluasi dengan cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(stacking_clf, X_train, y_train, cv=cv, scoring='accuracy')
        mean_score = cv_scores.mean()
        
        print(f"Stacking ensemble CV score: {mean_score:.4f}")
        
        return stacking_clf, mean_score
    
    def create_weighted_ensemble(self, X_train, y_train, X_test):
        """Create weighted ensemble based on CV scores"""
        print("Creating weighted ensemble...")
        
        # Get predictions from all models
        train_predictions = {}
        test_predictions = {}
        
        for name, model in self.optimized_models.items():
            model.fit(X_train, y_train)
            train_predictions[name] = model.predict_proba(X_train)
            test_predictions[name] = model.predict_proba(X_test)
        
        # Calculate weights based on CV scores with exponential scaling untuk memprioritaskan model terbaik
        scores = np.array(list(self.cv_scores.values()))
        exp_scores = np.exp(scores * 10)  # Eksponensial untuk memperkuat perbedaan
        total_score = sum(exp_scores)
        weights = {name: exp_scores[i]/total_score for i, name in enumerate(self.cv_scores.keys())}
        
        print("Model weights:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {weight:.4f}")
        
        # Weighted average of predictions
        weighted_train_pred = np.zeros_like(list(train_predictions.values())[0])
        weighted_test_pred = np.zeros_like(list(test_predictions.values())[0])
        
        for name, weight in weights.items():
            weighted_train_pred += weight * train_predictions[name]
            weighted_test_pred += weight * test_predictions[name]
        
        # Evaluasi weighted ensemble
        y_pred = np.argmax(weighted_train_pred, axis=1)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Weighted ensemble accuracy on training: {accuracy:.4f}")
        
        return weighted_train_pred, weighted_test_pred, weights
    
    def create_blending_ensemble(self, X_train, y_train, X_val, y_val, X_test):
        """Create blending ensemble using validation set"""
        print("Creating blending ensemble...")
        
        # Train models on training data
        val_predictions = {}
        test_predictions = {}
        
        for name, model in self.optimized_models.items():
            print(f"Training {name} for blending...")
            model.fit(X_train, y_train)
            val_predictions[name] = model.predict_proba(X_val)
            test_predictions[name] = model.predict_proba(X_test)
        
        # Create meta-features
        meta_features = np.hstack([pred for pred in val_predictions.values()])
        
        # Train meta-learner
        meta_learner = LogisticRegression(C=10, max_iter=2000, random_state=self.random_state)
        meta_learner.fit(meta_features, y_val)
        
        # Create test meta-features
        test_meta_features = np.hstack([pred for pred in test_predictions.values()])
        
        # Evaluate blending ensemble
        val_score = meta_learner.score(meta_features, y_val)
        print(f"Blending ensemble score on validation: {val_score:.4f}")
        
        return meta_learner, test_meta_features, val_score
    
    def fit_best_ensemble(self, X_train, y_train, X_test, X_val=None, y_val=None):
        """Fit and compare different ensemble methods"""
        print("Comparing ensemble methods...")
        
        # Create base models
        self.create_base_models()
        
        # Optimize hyperparameters
        self.optimize_hyperparameters(X_train, y_train)
        
        # Compare ensemble methods
        ensemble_scores = {}
        
        # Voting ensemble
        voting_clf, voting_score = self.create_voting_ensemble(X_train, y_train)
        ensemble_scores['voting'] = voting_score
        
        # Stacking ensemble
        stacking_clf, stacking_score = self.create_stacking_ensemble(X_train, y_train)
        ensemble_scores['stacking'] = stacking_score
        
        # Weighted ensemble
        _, _, weights = self.create_weighted_ensemble(X_train, y_train, X_test)
        ensemble_scores['weighted'] = max(self.cv_scores.values())  # Use best individual score as proxy
        
        # Blending ensemble jika validation set tersedia
        if X_val is not None and y_val is not None:
            blending_clf, blending_meta_features, blending_score = self.create_blending_ensemble(
                X_train, y_train, X_val, y_val, X_test
            )
            ensemble_scores['blending'] = blending_score
        
        # Select best ensemble
        best_method = max(ensemble_scores.items(), key=lambda x: x[1])
        print(f"\nBest ensemble method: {best_method[0]} with score: {best_method[1]:.4f}")
        
        if best_method[0] == 'voting':
            self.ensemble_model = voting_clf
        elif best_method[0] == 'stacking':
            self.ensemble_model = stacking_clf
        elif best_method[0] == 'blending' and X_val is not None:
            self.ensemble_model = {'meta_learner': blending_clf, 'base_models': self.optimized_models}
        else:
            # For weighted, we'll use the best individual model
            best_individual = max(self.cv_scores.items(), key=lambda x: x[1])
            self.ensemble_model = self.optimized_models[best_individual[0]]
        
        # Fit the best ensemble if not already fitted
        if best_method[0] not in ['weighted', 'blending']:
            self.ensemble_model.fit(X_train, y_train)
        
        return self.ensemble_model, best_method[1]
    
    def predict(self, X_test):
        """Make predictions using the best ensemble"""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model fitted. Call fit_best_ensemble first.")
        
        # Handle blending ensemble case
        if isinstance(self.ensemble_model, dict) and 'meta_learner' in self.ensemble_model:
            # Get predictions from base models
            test_predictions = {}
            for name, model in self.ensemble_model['base_models'].items():
                test_predictions[name] = model.predict_proba(X_test)
            
            # Create meta-features
            meta_features = np.hstack([pred for pred in test_predictions.values()])
            
            # Predict with meta-learner
            return self.ensemble_model['meta_learner'].predict(meta_features)
        else:
            return self.ensemble_model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        if self.ensemble_model is None:
            raise ValueError("No ensemble model fitted. Call fit_best_ensemble first.")
        
        # Handle blending ensemble case
        if isinstance(self.ensemble_model, dict) and 'meta_learner' in self.ensemble_model:
            # Get predictions from base models
            test_predictions = {}
            for name, model in self.ensemble_model['base_models'].items():
                test_predictions[name] = model.predict_proba(X_test)
            
            # Create meta-features
            meta_features = np.hstack([pred for pred in test_predictions.values()])
            
            # Predict with meta-learner
            return self.ensemble_model['meta_learner'].predict_proba(meta_features)
        else:
            return self.ensemble_model.predict_proba(X_test)
    
    def get_param_grids(self):
        """Define parameter grids for each model"""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [6, 8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None],
                'bootstrap': [True, False]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [6, 8, 10, 12, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0],
                'class_weight': ['balanced', None],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_samples': [5, 10, 20]
            },
            'catboost': {
                'iterations': [100, 200, 300],
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5, 7],
                'border_count': [32, 64, 128],
                'subsample': [0.6, 0.8, 1.0],
                'random_strength': [0.1, 1.0, 10.0],
                'grow_policy': ['SymmetricTree', 'Depthwise']
            },
            'logistic': {
                # Kombinasi untuk penalty=None
                'C': [1.0],  # C tidak berpengaruh ketika penalty=None
                'penalty': [None],
                'solver': ['newton-cg', 'lbfgs', 'sag'],
                'class_weight': ['balanced', None],
                'max_iter': [2000]
            },
            'logistic_l1': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [2000]
            },
            'logistic_l2': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [2000]
            },
            'logistic_elasticnet': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['elasticnet'],
                'solver': ['saga'],
                'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9],  # Lebih banyak opsi untuk elasticnet
                'class_weight': ['balanced', None],
                'max_iter': [2000]
            },
            # Tambahkan di bagian get_param_grids() untuk XGBoost
            'xgboost': {
                'n_estimators': [100, 200, 300, 500, 1000],
                'max_depth': [3, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'gamma': [0, 0.1, 0.5, 1.0],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0],
                'scale_pos_weight': [1, 3, 5]
                # Hapus early_stopping_rounds dan learning_rate_decay
            }
        }
        return param_grids

# Tambahkan di bagian import
import optuna
from sklearn.model_selection import cross_val_score

# Tambahkan di bagian import
from sklearn.calibration import CalibratedClassifierCV

def calibrate_models(self, X_train, y_train):
    """Calibrate model probabilities"""
    print("Calibrating model probabilities...")
    
    calibrated_models = {}
    
    for name, model in self.optimized_models.items():
        print(f"Calibrating {name}...")
        
        # Gunakan sigmoid calibration untuk model probabilistik
        calibrated = CalibratedClassifierCV(
            model, 
            method='sigmoid', 
            cv=5,
            n_jobs=-1
        )
        
        calibrated.fit(X_train, y_train)
        calibrated_models[name] = calibrated
    
    return calibrated_models

def objective(trial, model_class, X, y, cv):
    # Definisikan parameter space berdasarkan jenis model
    if model_class.__name__ == 'RandomForestClassifier':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
    elif model_class.__name__ == 'LogisticRegression':
        params = {
            'C': trial.suggest_float('C', 1e-5, 100, log=True),
            'solver': trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']),
            'max_iter': 2000,
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
    # Tambahkan definisi untuk model lain...
    
    # Buat model dengan parameter yang diusulkan
    model = model_class(**params)
    
    # Evaluasi model dengan cross-validation
    score = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    
    return score

# Tambahkan metode ini ke dalam class EnsembleOptimizer
def optimize_with_optuna(self, X_train, y_train, model_name, model_class, n_trials=100):
    """Optimize model hyperparameters using Optuna"""
    print(f"Optimizing {model_name} with Optuna...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
    
    # Buat study Optuna
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
    
    # Optimize
    study.optimize(lambda trial: objective(trial, model_class, X_train, y_train, cv), n_trials=n_trials)
    
    # Dapatkan parameter terbaik
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"Best {model_name} CV score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Buat model dengan parameter terbaik
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)
    
    return best_model, best_score

def create_multi_level_stacking(self, X_train, y_train):
    """Create multi-level stacking ensemble"""
    print("Creating multi-level stacking ensemble...")
    
    # Level 1 base models
    level1_estimators = [(name, model) for name, model in self.optimized_models.items()]
    
    # Level 1 meta-learner
    level1_meta = LogisticRegression(C=10, max_iter=2000, random_state=self.random_state)
    
    # Level 1 stacking
    level1_stacking = StackingClassifier(
        estimators=level1_estimators,
        final_estimator=level1_meta,
        cv=5,
        stack_method='auto',
        n_jobs=-1,
        passthrough=True
    )
    
    # Level 2 base models
    level2_estimators = [
        ('level1_stack', level1_stacking),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=10, random_state=self.random_state)),
        ('xgb', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=self.random_state))
    ]
    
    # Level 2 meta-learner
    level2_meta = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=self.random_state
    )
    
    # Level 2 stacking
    level2_stacking = StackingClassifier(
        estimators=level2_estimators,
        final_estimator=level2_meta,
        cv=5,
        stack_method='auto',
        n_jobs=-1
    )
    
    # Evaluasi dengan cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
    cv_scores = cross_val_score(level2_stacking, X_train, y_train, cv=cv, scoring='accuracy')
    mean_score = cv_scores.mean()
    
    print(f"Multi-level stacking ensemble CV score: {mean_score:.4f}")
    
    return level2_stacking, mean_score

def create_diverse_ensemble(self, X_train, y_train, lambda_param=0.1):
    """Create ensemble with diversity using Negative Correlation Learning"""
    print("Creating diverse ensemble with NCL...")
    
    # Buat beberapa model dasar dengan parameter berbeda
    base_models = []
    for i in range(10):  # Buat 10 model berbeda
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=np.random.randint(3, 15),
            min_samples_split=np.random.randint(2, 10),
            random_state=self.random_state + i
        )
        base_models.append(rf)
    
    # Train models sequentially with negative correlation
    trained_models = []
    for i, model in enumerate(base_models):
        print(f"Training diverse model {i+1}/10...")
        
        # Fit model
        model.fit(X_train, y_train)
        trained_models.append(model)
        
        if i > 0:
            # Calculate ensemble prediction so far
            ensemble_pred = np.zeros((X_train.shape[0], len(np.unique(y_train))))
            for trained_model in trained_models[:-1]:
                ensemble_pred += trained_model.predict_proba(X_train)
            ensemble_pred /= len(trained_models) - 1
            
            # Calculate diversity penalty
            current_pred = trained_models[-1].predict_proba(X_train)
            diversity_penalty = lambda_param * np.sum((current_pred - ensemble_pred) ** 2)
            
            print(f"Diversity penalty: {diversity_penalty:.4f}")
    
    # Create final ensemble
    voting_clf = VotingClassifier(
        estimators=[(f'model_{i}', model) for i, model in enumerate(trained_models)],
        voting='soft',
        n_jobs=-1
    )
    
    # Evaluate ensemble
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
    cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='accuracy')
    mean_score = cv_scores.mean()
    
    print(f"Diverse ensemble CV score: {mean_score:.4f}")
    
    return voting_clf, mean_score