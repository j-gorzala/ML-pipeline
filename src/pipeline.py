from src.space import xgb_classifier_space, xgb_regressor_space

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, \
                            mean_squared_error, \
                            roc_curve, \
                            confusion_matrix, \
                            classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier, \
                    XGBRegressor
from pandas_profiling import ProfileReport
from BorutaShap import BorutaShap
from umap.umap_ import UMAP
from collections import Counter

import yaml
import csv
import pickle
import hyperopt
import shap
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os




class MLPipeline:


    def read_params(self):
        print(f"\n{20 * '_'}Read params node")
        with open('params/parameters.yaml', 'r') as f:
            self.params = yaml.load(f, Loader=yaml.SafeLoader)


    def read_data(self):
        print(f"\n{20 * '_'}Read data node")
        input_data_path = self.params['paths']['input_data']
        data_train_filename = self.params['sample_params']['data_train_filename']
        data_test_filename = self.params['sample_params']['data_test_filename']
        
        self.data_train = pd.read_csv(f"{input_data_path}/{data_train_filename}.csv", index_col=False) \
                            .reset_index(drop=True)
        self.data_test = pd.read_csv(f"{input_data_path}/{data_test_filename}.csv", index_col=False) \
                            .reset_index(drop=True)


    def create_eda_reports(self):
        print(f"\n{20 * '_'}EDA reports node")
        eda_reports_path = self.params['paths']['eda_reports']
        # Train report
        profile_train = ProfileReport(self.data_train, title="Pandas Profiling Report - Train", minimal=True)
        profile_train.to_file(f"{eda_reports_path}/data_report_train.html")
        # Test report
        profile_test = ProfileReport(self.data_test, title="Pandas Profiling Report - Test", minimal=True)
        profile_test.to_file(f"{eda_reports_path}/data_report_test.html")


    def data_preprocessing(self):
        print(f"\n{20 * '_'}Data preprocessing node")
        target_col = self.params['data_preprocessing_params']['target_column']
        cols_to_drop = self.params['data_preprocessing_params']['drop_columns']
        imputer_k_value = self.params['data_preprocessing_params']['imputer_k_value']
        preprocessing_path = self.params['paths']['preprocessing']
        standard_scaling = self.params['data_preprocessing_params']['standard_scaling']

        # X, y split
        self.X_train_val = self.data_train.drop(columns=[target_col])
        self.y_train_val = self.data_train[[target_col]].values
        if target_col in self.data_test.columns:
            self.X_test = self.data_test.drop(columns=[target_col])
            self.y_test = self.data_test[[target_col]].values
        else:
            self.X_test = self.data_test

        # Drop unused columns
        if len(cols_to_drop) > 0:
            self.X_train_val = self.X_train_val.drop(columns=cols_to_drop)
            self.X_test = self.X_test.drop(columns=cols_to_drop)

        # Dummy encoding
        X_train_n_rows = self.X_train_val.shape[0]
        X_data = pd.concat([self.X_train_val, self.X_test], ignore_index=True).reset_index(drop=True)
        cat_cols = X_data.select_dtypes(include=['category', 'object']).columns.tolist()
        X_data = pd.get_dummies(X_data, columns=cat_cols)
        self.X_train_val = X_data.iloc[:X_train_n_rows, :]
        self.X_test = X_data.iloc[X_train_n_rows:, :]

        # Standard scaling
        if standard_scaling:
            scaler = StandardScaler()
            scaler.fit(self.X_train_val)
            # X_train_val
            X_train_val_scaled = scaler.transform(self.X_train_val)
            self.X_train_val = pd.DataFrame(X_train_val_scaled, index=self.X_train_val.index, columns=self.X_train_val.columns)
            # X_test
            X_test_scaled = scaler.transform(self.X_test)
            self.X_test = pd.DataFrame(X_test_scaled, index=self.X_test.index, columns=self.X_test.columns)
            
        # Missing values imputation
        if self.params['data_preprocessing_params']['imputer']:
            imputer = KNNImputer(n_neighbors=imputer_k_value)
            # X_train_val
            X_train_val_imputed = imputer.fit_transform(self.X_train_val.values)
            self.X_train_val = pd.DataFrame(X_train_val_imputed, index=self.X_train_val.index, columns=self.X_train_val.columns)
            # X_test
            X_test_imputed = imputer.transform(self.X_test.values)
            self.X_test = pd.DataFrame(X_test_imputed, index=self.X_test.index, columns=self.X_test.columns)

        # Clustering
        if self.params['data_preprocessing_params']['clustering']:
            n_clusters = self.params['data_preprocessing_params']['n_clusters']
            if self.params['data_preprocessing_params']['imputer']:
                X_train_val_clust = self.X_train_val
                X_test_clust = self.X_test
            else:
                imputer = KNNImputer(n_neighbors=imputer_k_value)
                # X_train_val
                X_train_val_imputed = imputer.fit_transform(self.X_train_val.values)
                X_train_val_clust = pd.DataFrame(X_train_val_imputed, index=self.X_train_val.index, columns=self.X_train_val.columns)
                # X_test
                X_test_imputed = imputer.transform(self.X_test.values)
                X_test_clust = pd.DataFrame(X_test_imputed, index=self.X_test.index, columns=self.X_test.columns)

            # Create the UMAP object
            umap_model = UMAP(random_state=42)
            # Fit the UMAP model to the training data
            umap_model.fit(X_train_val_clust)
            
            # Use the UMAP model to transform the training and test data
            X_train_val_reduced = umap_model.transform(X_train_val_clust)
            X_test_reduced = umap_model.transform(X_test_clust)
            
            # Create a KMeans model and get the labels
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X_train_val_reduced)
            labels_train_val = kmeans.predict(X_train_val_reduced)
            labels_test = kmeans.predict(X_test_reduced)

            # Labels counts
            print(f'\nTrain clusters size: {Counter(labels_train_val)}')
            print(f'Test clusters size: {Counter(labels_test)}')

            # Create a scatter plot of the transformed train data points
            plt.scatter(X_train_val_reduced[:, 0], X_train_val_reduced[:, 1], c=labels_train_val)
            plt.title("UMAP Plot - train")
            plt.savefig(f"{preprocessing_path}/clustering_plot_train_val.png")
            plt.clf()

            # Create a scatter plot of the transformed test data points
            plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=labels_test)
            plt.title("UMAP Plot - test")
            plt.savefig(f"{preprocessing_path}/clustering_plot_test.png")
            plt.clf()

            # Save clustering labels
            self.X_train_val['clust_label'] = labels_train_val
            self.X_test['clust_label'] = labels_test


    def _feature_selection(self, X_train_val, y_train_val, clust):
        print(f"\n{20 * '_'}Feature selection node")
        boruta_shap_n_trials = self.params['feature_selection_params']['boruta_shap_n_trials']
        use_clustering = self.params['feature_selection_params']['use_clustering']
        ds_problem = self.params['global_params']['problem']
        feature_selection_path = self.params['paths']['feature_selection']
        
        if use_clustering:
            # Drop clustering label
            X_train_val = X_train_val.drop(columns=['clust_label'])

        if ds_problem == 'classification':
            model = XGBClassifier(scale_pos_weight=np.mean(y_train_val))
            Feature_Selector = BorutaShap(model=model,
                                        importance_measure='shap',
                                        classification=True)
            Feature_Selector.fit(X=X_train_val, 
                                y=y_train_val, 
                                n_trials=boruta_shap_n_trials, 
                                random_state=42)
            Feature_Selector.TentativeRoughFix()
        
        elif ds_problem == 'regression':
            model = XGBRegressor()
            Feature_Selector = BorutaShap(model=model,
                                          importance_measure='shap',
                                          classification=False)
            Feature_Selector.fit(X=X_train_val, 
                                 y=y_train_val, 
                                 n_trials=boruta_shap_n_trials, 
                                 random_state=42)
            Feature_Selector.TentativeRoughFix()
        
        # Display Feature Selector features
        plt = Feature_Selector.plot(X_size=12, 
                                figsize=(12,8),
                                y_scale='log', 
                                which_features='all')
        
        # Save to .csv
        Feature_Selector.results_to_csv(filename=f'{feature_selection_path}/feature_importance_clust{clust}')


    def feature_selection(self):
        use_clustering = self.params['feature_selection_params']['use_clustering']
        if use_clustering:
            n_clusters = self.params['data_preprocessing_params']['n_clusters']
            for clust in range(0, n_clusters):
                X_train_val = self.X_train_val[self.X_train_val['clust_label']==clust]
                y_train_val = self.y_train_val[X_train_val.index]
                self._feature_selection(X_train_val, y_train_val, clust)
        else:
            self._feature_selection(self.X_train_val, self.y_train_val, 0)


    def _create_basic_model(self, X_train_val, y_train_val, clust):
        print(f"\n{20 * '_'}Basic model node")
        shortlist_path = self.params['paths']['feature_selection']
        model_path = self.params['paths']['model']
        ds_problem = self.params['global_params']['problem']
        data_science_params = self.params['data_science_params']
        use_shortlist = data_science_params['use_shortlist']
        use_clustering = data_science_params['use_clustering']
        cv_n_splits = data_science_params['cv_n_splits']
        overfit_penalty_factor = data_science_params['overfit_penalty_factor']
        crossval_penalty_factor = data_science_params['crossval_penalty_factor']

        if use_clustering:
            # Drop clustering label
            X_train_val = X_train_val.drop(columns=['clust_label'])

        if use_shortlist and 'feature_importance.csv' in os.listdir(shortlist_path):
            feature_importance = pd.read_csv(f'{shortlist_path}/feature_importance_clust{clust}.csv')
            shortlist = feature_importance.loc[feature_importance['Decision'] == 'Accepted', 'Features'].tolist()
            X_train_val = X_train_val[shortlist]
        
        # CV Folds
        cv = KFold(n_splits=cv_n_splits, shuffle=True)
        
        if ds_problem == 'classification':
            objective = 'binary:logistic'
            model = XGBClassifier(
                        objective=objective,
                        scale_pos_weight=np.mean(y_train_val),
                        booster=data_science_params['booster'],
                        verbosity=data_science_params['verbosity'],
                        nthread=data_science_params['nthread'],
                        eta=data_science_params['eta'],
                        max_depth=data_science_params['max_depth'],
                        min_child_weight=data_science_params['min_child_weight'],
                        subsample=data_science_params['subsample'],
                        colsample_bytree=data_science_params['colsample_bytree'],
                        early_stopping_rounds=data_science_params['early_stopping_rounds']
                    )
            # Use cross-validation with early stopping to evaluate the model
            scores = []
            for train_index, test_index in cv.split(X_train_val):
                model.fit(X_train_val.iloc[train_index,:], 
                            y_train_val[train_index], 
                            verbose=0,
                            eval_set=[(X_train_val.iloc[test_index, :], y_train_val[test_index])])
                acc_train = accuracy_score(y_train_val[train_index], model.predict(X_train_val.iloc[train_index, :]))     
                acc_test = accuracy_score(y_train_val[test_index], model.predict(X_train_val.iloc[test_index, :]))
                scores.append(acc_test - overfit_penalty_factor * np.abs(acc_train - acc_test))
            score = np.mean(scores) - crossval_penalty_factor * (np.max(scores) - np.min(scores))
            print(f'\nCV penalized scores (Accuracy): {scores}')
            print(f'CV penalized mean score (Accuracy): {score:.2f}')
        
        elif ds_problem == 'regression':
            objective = 'reg:squarederror'
            model = XGBRegressor(
                        objective=objective,
                        booster=data_science_params['booster'],
                        verbosity=data_science_params['verbosity'],
                        nthread=data_science_params['nthread'],
                        eta=data_science_params['eta'],
                        max_depth=data_science_params['max_depth'],
                        min_child_weight=data_science_params['min_child_weight'],
                        subsample=data_science_params['subsample'],
                        colsample_bytree=data_science_params['colsample_bytree']
                        )
            # Use cross-validation with early stopping to evaluate the model
            scores = []
            for train_index, test_index in cv.split(X_train_val):
                model.fit(X_train_val.iloc[train_index,:], 
                            y_train_val[train_index], 
                            verbose=0,
                            eval_set=[(X_train_val.iloc[test_index, :], y_train_val[test_index])])
                mse_train = mean_squared_error(y_train_val[train_index], model.predict(X_train_val.iloc[train_index, :]), squared=False)     
                mse_test = mean_squared_error(y_train_val[test_index], model.predict(X_train_val.iloc[test_index, :]), squared=False)
                scores.append(mse_test + overfit_penalty_factor * np.abs(mse_train - mse_test))
            score = np.mean(scores) + crossval_penalty_factor * (np.max(scores) - np.min(scores))
            print(f'\nCV penalized scores (MSE): {scores}')
            print(f'CV penalized mean score (MSE): {score:.2f}')

        # Save the model to a file
        setattr(model, 'model_variables', X_train_val.columns.tolist())
        with open(f'{model_path}/basic_model_clust{clust}.pkl', 'wb') as file:
            pickle.dump(model, file)


    def create_basic_model(self):
        use_clustering = self.params['data_science_params']['use_clustering']
        if use_clustering:
            n_clusters = self.params['data_preprocessing_params']['n_clusters']
            for clust in range(0, n_clusters):
                X_train_val = self.X_train_val[self.X_train_val['clust_label']==clust]
                y_train_val = self.y_train_val[X_train_val.index]
                self._create_basic_model(X_train_val, y_train_val, clust)
        else:
            self._create_basic_model(self.X_train_val, self.y_train_val, 0)


    def _hyperopt(self, X_train_val, y_train_val, clust):        
        print(f"\n{20 * '_'}Hyperopt node")
        def _objective(params):
            # CV Folds
            cv = KFold(n_splits=cv_n_splits, shuffle=True)
            
            if ds_problem == 'classification':
                model = XGBClassifier(**params)
                # Use cross-validation with early stopping to evaluate the model
                scores = []
                for train_index, test_index in cv.split(X_train_val):
                    model.fit(X_train_val.iloc[train_index,:], 
                                y_train_val[train_index], 
                                verbose=0,
                                eval_set=[(X_train_val.iloc[test_index, :], y_train_val[test_index])])
                    acc_train = accuracy_score(y_train_val[train_index], model.predict(X_train_val.iloc[train_index, :]))     
                    acc_test = accuracy_score(y_train_val[test_index], model.predict(X_train_val.iloc[test_index, :]))
                    scores.append(acc_test - overfit_penalty_factor * np.abs(acc_train - acc_test))
                score = -np.mean(scores) + crossval_penalty_factor * (np.max(scores) - np.min(scores))
            
            elif ds_problem == 'regression':
                model = XGBRegressor(**params)
                # Use cross-validation with early stopping to evaluate the model
                scores = []
                for train_index, test_index in cv.split(X_train_val):
                    model.fit(X_train_val.iloc[train_index,:], 
                                y_train_val[train_index], 
                                verbose=0,
                                eval_set=[(X_train_val.iloc[test_index, :], y_train_val[test_index])])
                    mse_train = mean_squared_error(y_train_val[train_index], model.predict(X_train_val.iloc[train_index, :]), squared=False)     
                    mse_val = mean_squared_error(y_train_val[test_index], model.predict(X_train_val.iloc[test_index, :]), squared=False)
                    scores.append(mse_val + overfit_penalty_factor * np.abs(mse_train - mse_val))
                score = np.mean(scores) + crossval_penalty_factor * (np.max(scores) - np.min(scores))

            return score
        
        #current_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        ds_problem = self.params['global_params']['problem']
        shortlist_path = self.params['paths']['feature_selection']
        model_path = self.params['paths']['model']
        hyperopt_path = self.params['paths']['hyperopt']
        hyperopt_params = self.params['hyperopt_params']
        use_shortlist = hyperopt_params['use_shortlist']
        use_clustering = hyperopt_params['use_clustering']
        cv_n_splits = hyperopt_params['cv_n_splits']
        overfit_penalty_factor = hyperopt_params['overfit_penalty_factor']
        crossval_penalty_factor = hyperopt_params['crossval_penalty_factor']
        
        if use_clustering:
            # Drop clustering label
            X_train_val = X_train_val.drop(columns=['clust_label'])

        if use_shortlist and 'feature_importance.csv' in os.listdir(shortlist_path):
            feature_importance = pd.read_csv(f'{shortlist_path}/feature_importance_clust{clust}.csv')
            shortlist = feature_importance.loc[feature_importance['Decision'] == 'Accepted', 'Features'].tolist()
            X_train_val = X_train_val[shortlist]
        
        if ds_problem == 'classification':
            xgb_space = xgb_classifier_space
        elif ds_problem == 'regression':
            xgb_space = xgb_regressor_space

        trials = hyperopt.Trials()
        best = hyperopt.fmin(fn=_objective,
                             space=xgb_space, 
                             algo=hyperopt.tpe.suggest,
                             max_evals=hyperopt_params['hyperopt_max_evals'],
                             trials=trials)
        
        # Write the log to a CSV file
        with open(f'{hyperopt_path}/log_clust{clust}.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=trials.trials[0]['result'].keys())
            writer.writeheader()
            for trial in trials.trials:
                result = trial['result']
                result['misc'] = str(trial['misc'])
                writer.writerow(result)

        # Train the XGBoost model with the best hyperparameters
        best_params = hyperopt.space_eval(xgb_space, best)
        print('\n')
        print(best_params)
        
        # Save best model
        # CV Folds
        cv = KFold(n_splits=cv_n_splits, shuffle=True)
        if ds_problem == 'classification':
            model = XGBClassifier(**best_params)
            scores = []
            for train_index, test_index in cv.split(X_train_val):
                model.fit(X_train_val.iloc[train_index,:], 
                            y_train_val[train_index], 
                            verbose=0,
                            eval_set=[(X_train_val.iloc[test_index, :], y_train_val[test_index])])
                acc_train = accuracy_score(y_train_val[train_index], model.predict(X_train_val.iloc[train_index, :]))     
                acc_test = accuracy_score(y_train_val[test_index], model.predict(X_train_val.iloc[test_index, :]))
                scores.append(acc_test - overfit_penalty_factor * np.abs(acc_train - acc_test))
            score = np.mean(scores) - crossval_penalty_factor * (np.max(scores) - np.min(scores))
            print(f'\nHyperopt CV folds scores: {scores}')
            print(f'Hyperopt CV accuracy: {score}')

        elif ds_problem == 'regression':
            model = XGBRegressor(**best_params)
            scores = []
            for train_index, test_index in cv.split(X_train_val):
                model.fit(X_train_val.iloc[train_index,:], 
                            y_train_val[train_index], 
                            verbose=0,
                            eval_set=[(X_train_val.iloc[test_index, :], y_train_val[test_index])])
                mse_train = mean_squared_error(y_train_val[train_index], model.predict(X_train_val.iloc[train_index, :]), squared=False)     
                mse_test = mean_squared_error(y_train_val[test_index], model.predict(X_train_val.iloc[test_index, :]), squared=False)
                scores.append(mse_test + overfit_penalty_factor * np.abs(mse_train - mse_test))
            score = np.mean(scores)  + crossval_penalty_factor * (np.max(scores) - np.min(scores))
            print(f'\nHyperopt CV folds: {scores}')
            print(f'Hyperopt mean CV MSE: {score}')

        # Save the model to a file
        setattr(model, 'model_variables', X_train_val.columns.tolist())
        with open(f'{model_path}/hyperopt_model_clust{clust}.pkl', 'wb') as file:
            pickle.dump(model, file)


    def hyperopt(self):
        use_clustering = self.params['hyperopt_params']['use_clustering']
        if use_clustering:
            n_clusters = self.params['data_preprocessing_params']['n_clusters']
            for clust in range(0, n_clusters):
                X_train_val = self.X_train_val[self.X_train_val['clust_label']==clust]
                y_train_val = self.y_train_val[X_train_val.index]
                self._hyperopt(X_train_val, y_train_val, clust)
        else:
            self._hyperopt(self.X_train_val, self.y_train_val, 0)


    def make_prediction(self):
        print(f"\n{20 * '_'}Prediction node")
        prediction_path = self.params['paths']['prediction']
        model_path = self.params['paths']['model']
        model_name = self.params['prediction_params']['model_name']
        target_col = self.params['data_preprocessing_params']['target_column']
        index_col = self.params['data_preprocessing_params']['index_column']
        use_clustering = self.params['prediction_params']['use_clustering']
        
        if use_clustering:
            clust_predictions = pd.DataFrame()
            n_clusters = self.params['data_preprocessing_params']['n_clusters']
            # Load the model from the file
            for clust in range(0, n_clusters):
                with open(f'{model_path}/{model_name}_clust{clust}.pkl', 'rb') as file:
                    model = pickle.load(file)
                # Use the loaded model to make predictions
                X_test = self.X_test[self.X_test['clust_label']==clust][model.model_variables]
                test_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=['test_pred'])
                clust_predictions = clust_predictions.append(test_pred)
            clust_predictions = clust_predictions.sort_index()
            # Save prediction with index
            out_dict = {}
            if index_col != '':
                out_dict[index_col] = self.data_test[index_col]
            out_dict[target_col] = clust_predictions['test_pred'].values
            predictions = pd.DataFrame(out_dict)
            predictions.to_csv(f'{prediction_path}/{model_name}_all_clust_predictions.csv', index=0)
        
        else:
            # Load the model from the file
            with open(f'{model_path}/{model_name}_clust0.pkl', 'rb') as file:
                model = pickle.load(file)
            
            # Use the loaded model to make predictions
            test_pred = model.predict(self.X_test[model.model_variables])
            
            # Save prediction with index
            out_dict = {}
            if index_col != '':
                out_dict[index_col] = self.data_test[index_col]
            out_dict[target_col] = test_pred
            predictions = pd.DataFrame(out_dict)
            predictions.to_csv(f'{prediction_path}/{model_name}_clust0_predictions.csv', index=0)


    def validation(self):
        print(f"\n{20 * '_'}Validation node")
        validation_path = self.params['paths']['validation']
        model_path = self.params['paths']['model']
        model_name = self.params['prediction_params']['model_name']
        target_col = self.params['data_preprocessing_params']['target_column']
        ds_problem = self.params['global_params']['problem']
        
        # Load the model from the file
        with open(f'{model_path}/{model_name}_clust0.pkl', 'rb') as file:
            model = pickle.load(file)
            model_variables = model.model_variables
        
        X_train_val = self.X_train_val[model_variables]

        # compute SHAP values
        explainer = shap.Explainer(model, X_train_val)
        shap_values = explainer(X_train_val)
        # Create the SHAP plot
        shap.summary_plot(shap_values, show=False, plot_size=(15, 8))
        plt.savefig(f"{validation_path}/{model_name}_clust0_shap_plot.png")
        plt.clf()

        # Create plots with metrices if target column is in test data
        if target_col in self.data_test.columns:
            
            if ds_problem == 'classification':
                def _calc_lift(predictions, labels):
                    """Calculate the lift for a set of predictions.
                    Parameters:
                        predictions (array-like): An array of predictions.
                        labels (array-like): An array of labels.
                    Returns:
                        lift (list): A list of lift values for each prediction.
                    """
                    # Create a list to store the lift values
                    lift = []
                    # Sort the predictions and labels by the predictions in descending order
                    sorted_predictions, sorted_labels = zip(*sorted(zip(predictions, labels), reverse=True))
                    # Calculate the number of positive and negative examples
                    num_pos = sum(labels)
                    num_neg = len(labels) - num_pos
                    # Initialize the running totals
                    running_pos = 0
                    running_neg = 0
                    # Iterate through the sorted predictions and labels
                    for pred, label in zip(sorted_predictions, sorted_labels):
                        # Increment the running totals
                        if label == 1:
                            running_pos += 1
                        else:
                            running_neg += 1
                        # Calculate the lift for this prediction
                        lift.append((running_pos / num_pos) / (running_neg / num_neg))
                    return lift

                # Make predictions on the training and test sets
                train_pred = model.predict_proba(X_train_val[model.model_variables])[:,1]
                test_pred = model.predict_proba(self.X_test[model.model_variables])[:,1]

                # Calculate the TPR and FPR for the training and test sets
                fpr_train, tpr_train, _ = roc_curve(self.y_train_val, train_pred)
                fpr_test, tpr_test, _ = roc_curve(self.y_test, test_pred)
                # Plot the ROC curve for the training and test sets
                plt.plot(fpr_train, tpr_train, label="Training set")
                plt.plot(fpr_test, tpr_test, label="Test set")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.title("ROC Plot")
                plt.savefig(f"{validation_path}/{model_name}_clust0_roc_plot.png")
                plt.clf()

                # Calculate the lift for the training and test sets
                lift_train = _calc_lift(model.predict(X_train_val[model.model_variables]), self.y_train_val)
                lift_test = _calc_lift(model.predict(self.X_test[model.model_variables]), self.y_test)
                # Plot the lift for the training and test sets on the same plot
                plt.plot(lift_train, label="Training set")
                plt.plot(lift_test, label="Test set")
                plt.legend()
                plt.title("Lift Plot")
                plt.savefig(f"{validation_path}/{model_name}_clust0_lift_plot.png")
                plt.clf()

                # Calculate the classification report and confusion matrix 
                print('\nClassification report train:')
                print(classification_report(self.y_train_val, model.predict(X_train_val[model.model_variables])))
                confusion_matrix_train = confusion_matrix(self.y_train_val, model.predict(X_train_val[model.model_variables]))
                print('Train set confusion matrix:')
                print(confusion_matrix_train)
                
                print('\nClassification report test:')
                print(classification_report(self.y_test, model.predict(self.X_test[model.model_variables])))
                confusion_matrix_test = confusion_matrix(self.y_test, model.predict(self.X_test[model.model_variables]))
                print('Test set confusion matrix:')
                print(confusion_matrix_test)
            
            elif ds_problem == 'regression':
                # Make predictions on the training and test sets
                train_pred = model.predict(X_train_val[model.model_variables])
                test_pred = model.predict(self.X_test[model.model_variables])

                # Calculate the RMSE
                train_rmse = mean_squared_error(self.y_train_val, train_pred, squared=False)
                test_rmse = mean_squared_error(self.y_test, test_pred, squared=False)
                # Create a bar plot showing the RMSE on the train and test sets
                plt.bar(["Train", "Test"], [train_rmse, test_rmse])
                plt.xlabel("Set")
                plt.ylabel("RMSE")
                plt.title("Root Mean Squared Error")
                plt.savefig(f"{validation_path}/{model_name}_clust0_rmse_plot.png")
                plt.clf()

                # Calculate the MAPE
                train_mape = np.mean(np.abs((self.y_train_val - train_pred) / self.y_train_val)) * 100
                test_mape = np.mean(np.abs((self.y_test - test_pred) / self.y_test)) * 100
                # Create a bar plot showing the MAPE on the train and test sets
                plt.bar(["Train", "Test"], [train_mape, test_mape])
                plt.xlabel("Set")
                plt.ylabel("MAPE")
                plt.title("Mean Absolute Percentage Error")
                plt.savefig(f"{validation_path}/{model_name}_clust0_mape_plot.png")
                plt.clf()