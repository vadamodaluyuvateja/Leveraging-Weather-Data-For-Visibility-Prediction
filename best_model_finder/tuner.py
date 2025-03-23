from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import lightgbm as lgb

class Model_Finder:
    """
    This class shall be used to find the model with the best accuracy and R2 score.
    """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.lightgbm = lgb.LGBMRegressor()

    def get_best_params_for_lightgbm(self, train_x, train_y):
        """
        Method Name: get_best_params_for_lightgbm
        Description: Get the parameters for LightGBM Algorithm which give the best accuracy.
                     Use Hyperparameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_lightgbm method of the Model_Finder class')
        try:
            # Defining the parameter grid
            self.param_grid_lgbm = {
                'boosting_type': ['gbdt', 'dart'],
                'num_leaves': [20, 31, 50],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.lightgbm, param_grid=self.param_grid_lgbm, cv=5, verbose=3)

            # Finding the best parameters
            self.grid.fit(train_x, train_y)

            # Extracting the best parameters
            self.best_params = self.grid.best_params_

            # Creating a new model with the best parameters
            self.lightgbm = lgb.LGBMRegressor(
                boosting_type=self.best_params['boosting_type'],
                num_leaves=self.best_params['num_leaves'],
                learning_rate=self.best_params['learning_rate'],
                n_estimators=self.best_params['n_estimators'],
                max_depth=self.best_params['max_depth'],
                subsample=self.best_params['subsample'],
                colsample_bytree=self.best_params['colsample_bytree']
            )

            # Training the new model
            self.lightgbm.fit(train_x, train_y)

            self.logger_object.log(self.file_object, f'LightGBM best params: {str(self.best_params)}. Exited the get_best_params_for_lightgbm method of the Model_Finder class')
            return self.lightgbm

        except Exception as e:
            self.logger_object.log(self.file_object, f'Exception occurred in get_best_params_for_lightgbm method of the Model_Finder class. Exception message: {str(e)}')
            self.logger_object.log(self.file_object, 'LightGBM Parameter tuning failed. Exited the get_best_params_for_lightgbm method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best R2 score.
        Output: The best model name and the model object
        On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_model method of the Model_Finder class')
        try:
            # Create best model for LightGBM
            self.lightgbm = self.get_best_params_for_lightgbm(train_x, train_y)
            self.prediction_lightgbm = self.lightgbm.predict(test_x)  # Predictions using the LightGBM Model
            self.lightgbm_r2_score = r2_score(test_y, self.prediction_lightgbm)

            self.logger_object.log(self.file_object, f'LightGBM R2 score: {self.lightgbm_r2_score}')

            return 'LightGBM', self.lightgbm

        except Exception as e:
            self.logger_object.log(self.file_object, f'Exception occurred in get_best_model method of the Model_Finder class. Exception message: {str(e)}')
            self.logger_object.log(self.file_object, 'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
