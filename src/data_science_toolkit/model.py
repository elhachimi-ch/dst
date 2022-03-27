"""
Under MIT License by EL HACHIMI CHOUAIB
"""
import tensorflow.keras.losses
import tensorflow.keras.optimizers
from tensorflow.keras import backend as K
import sklearn.tree as tree
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool1D, Conv1D, Reshape, LSTM
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from math import floor, ceil
from collections import deque
from math import sqrt
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.naive_bayes import GaussianNB
import graphviz
import tensorflow as tf
import numpy as np
from chart import Chart
from dataframe import DataFrame
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier



class Model:
    def __init__(
        self, 
        data_x=None, 
        data_y=None, 
        model_type='knn', 
        task='c',
        training_percent=1, 
        epochs=50, 
        batch_size=32, 
        generator=None,
        validation_percentage=0.2
        ):
        
        if training_percent != 1:
            self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(data_x, 
                                                                                            data_y,
                                                                                            train_size=training_percent,
                                                                                            test_size=1-training_percent)
        else:
            self.x = data_x
            self.y = data_y
        
        self.x = data_x
        self.y = data_y
        self.__y_pred = None
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__model_type = model_type
        self.__boosted_model = None
        self.__generator = generator
        self.history = 'None'
        self.__c_or_r_ts = task
        self.__validation_percentage = validation_percentage
        
        if model_type == 'dt':
            if task == 'c':
                self.__model = tree.DecisionTreeClassifier()
            else:
                self.__model = tree.DecisionTreeRegressor()

        elif model_type == 'svm':
            if task == 'c':
                self.__model = svm.SVC()
            else:
                self.__model = svm.SVR()
                
        elif model_type == 'lr':
            if task == 'c':
                self.__model = LogisticRegression(random_state=2)
            else:
                self.__model = LinearRegression()

        elif model_type == 'nb':
            if task == 'c':
                self.__model = MultinomialNB()
            else:
                self.__model = GaussianNB()
        
        elif model_type == 'rf':
            if task == 'c':
                self.__model = RandomForestClassifier()
            else:
                self.__model = RandomForestRegressor()
                
        elif model_type == 'xb':
            if task == 'c':
                self.__model = XGBClassifier()
            else:
                self.__model = XGBRegressor()

        elif model_type == 'dl':
            self.__model = Sequential()

        elif model_type == 'knn':
            if task == 'c':
                self.__model = KNeighborsClassifier(n_neighbors=5)
            else:
                self.__model = KNeighborsRegressor(n_neighbors=5)
                
        elif model_type == 'gb':
            if task == 'c':
                self.__model = GradientBoostingClassifier()
            else:
                self.__model = GradientBoostingRegressor() 
            
        else:
            self.__model = None

    def get_generator(self):
        return self.__generator
    
    def set_generator(self, generator):
        self.__generator = generator
        
    def get_model(self):
        return self.__model
    
    def set_model(self, model):
        self.__model = model
    
    def add_layer(self, connections_number=2, activation_function='relu', input_dim=None):
        """Add a dense layer to the model architecture

        Args:
            connections_number (int, optional): number of neurons to add. Defaults to 2.
            activation_function (str, optional): function to apply on sum of wi.xi. examples: ['linear', 'relu', 'softmax']. Defaults to 'relu'.
            input_dim (int, optional): number of features in X matrix. Defaults to None.
        """
        if input_dim:
            self.__model.add(Dense(connections_number, activation=activation_function, input_dim=input_dim))
        else:
            self.__model.add(Dense(connections_number, activation=activation_function))
            
    def add_lstm_layer(self, connections_number=2, activation_function='relu', input_shape=None, return_sequences=True):
        """Add a lstm layer

        Args:
            connections_number (int, optional): [description]. Defaults to 2.
            activation_function (str, optional): [description]. Defaults to 'relu'.
            input_shape ([type], optional): example: (weather_window,1). Defaults to None.
        """
        if input_shape is not None:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, input_shape=input_shape, return_sequences=return_sequences))
        else:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, return_sequences=return_sequences))

    def add_conv_2d_layer(self, filter_nbr=1, filter_shape_tuple=(3,3), input_shape=None, activation_function='relu'):
        if input_shape:
            self.__model.add(Conv2D(filters=filter_nbr, kernel_size=filter_shape_tuple, input_shape=input_shape,
                                    activation=activation_function))
        else:
            self.__model.add(Conv2D(filters=filter_nbr, kernel_size=filter_shape_tuple,
                                    activation=activation_function))
            
    def add_conv_1d_layer(self, filter_nbr=1, filter_shape_int=3, input_shape=None, activation_function='relu', strides=10):
        if input_shape:
            #Input size should be (n_features, 1) == (data_x.shape[1], 1)
            self.__model.add(Conv1D(filters=filter_nbr, kernel_size=filter_shape_int, input_shape=input_shape,
                                    activation=activation_function))
        else:
            self.__model.add(Conv1D(filters=filter_nbr, kernel_size=filter_shape_int,
                                    activation=activation_function))

    def add_pooling_2d_layer(self, pool_size_tuple=(2, 2)):
        self.__model.add(MaxPooling2D(pool_size=pool_size_tuple))

    def add_pooling_1d_layer(self, pool_size_int=2):
        self.__model.add(MaxPool1D(pool_size=pool_size_int))

    def add_flatten_layer(self):
        self.__model.add(Flatten())
        
    def add_reshape_layer(self, input_dim):
        """
        for 1dcnn and 2dcnn use this layer as first layer 
        """
        self.__model.add(Reshape((input_dim, 1), input_shape=(input_dim, )))

    """def add_reshape_layer(self, target_shape=None, input_shape=None):
        self.__model.add(Reshape(target_shape=target_shape, input_shape=input_shape))"""

    def add_dropout_layer(self, rate_to_keep_output_value=0.2):
        """ dropout default initial value """
        self.__model.add(Dropout(rate_to_keep_output_value))

    def train(self, loss=tensorflow.keras.losses.mse, optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), 
              metrics_as_list_of_functions_or_str=['accuracy']):
        
        """
        losses and metrics for regresion:
        tensorflow.keras.losses.mse
        r2_keras
        
        losses and metrics for classification:
        multi classes: tensorflow.keras.losses.categorical_crossentropy
        two classes: tensorflow.keras.losses.binary_crossentropy
                
        Optimizers:
        tensorflow.keras.optimizers.SGD(learning_rate=0.01)
        tensorflow.keras.optimizers.Adam(learning_rate=0.01)
        ...
        
        if you pass y as integers use loss='sparse_categorical_crossentropy'
        class Adadelta: Optimizer that implements the Adadelta algorithm.
        class Adagrad: Optimizer that implements the Adagrad algorithm.
        class Adam: Optimizer that implements the Adam algorithm.
        class Adamax: Optimizer that implements the Adamax algorithm.
        class Ftrl: Optimizer that implements the FTRL algorithm.
        class Nadam: Optimizer that implements the NAdam algorithm.
        class Optimizer: Base class for Keras optimizers.
        class RMSprop: Optimizer that implements the RMSprop algorithm.
        class SGD: Gradient descent (with momentum) optimizer.
        """
        if self.__model_type == 'dl':
            if 'r2_keras' in metrics_as_list_of_functions_or_str:
                metrics_as_list_of_functions_or_str.remove('r2_keras')
                metrics_as_list_of_functions_or_str.append(self.r2_keras)
            self.__model.compile(loss=loss, optimizer=optimizer, metrics=metrics_as_list_of_functions_or_str)
            if self.__generator is not None:
                self.history = self.__model.fit(self.get_generator(), epochs=self.__epochs, batch_size=self.__batch_size)
                print(self.history.history)
            else:
                self.history = self.__model.fit(self.x, self.y, epochs=self.__epochs,
                                        batch_size=self.__batch_size, validation_split=self.__validation_percentage)
                print(self.history.history)
                self.__y_pred = self.__model.predict(self.__x_test)
        else:
            self.__model.fit(self.__x_train, self.__y_train)
            self.__y_pred = self.__model.predict(self.__x_test)
        """history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']"""

    def summary(self):
        print(self.__model.summary())
        
    # banary classification
    def predict(self, x_to_pred):
        return self.__model.predict(x_to_pred)
    
    def forcast_next_step(self, window):
        current_batch = window.reshape((1, window.shape[0], 1))
        # One timestep ahead of historical 12 points
        return self.predict(current_batch)[0]

    def predict_proba(self, x_to_pred):
        return self.__model.predict_proba(x_to_pred)

    def accuracy(self):
        return accuracy_score(self.__y_test, self.__y_pred)

    def precision(self, binary_classification=False):
        if binary_classification:
            return precision_score(self.__y_test, self.__y_pred)
        return precision_score(self.__y_test, self.__y_pred, average=None)
    
    def recall(self):
        return recall_score(self.__y_test, self.__y_pred)

    def f1_score(self):
        return f1_score(self.__y_test, self.__y_pred)

    def regression_report(self, y_test=None, y_predicted=None, savefig=False): 
        """
        pass y_test and y_predected as pandas serie is get_column
        """
        
        if y_test is not None and y_predicted is not None:
            self.__y_test = y_test
            self.__y_pred = y_predicted
        
        data = DataFrame(self.__y_test, data_type='list', columns_names_as_list=['y_test'], data_types_in_order=[float])
        data.add_column(self.__y_pred, 'y_predicted')
        data.reset_index(drop=True)
        sns.set_theme(color_codes=True)
        x_plot = np.linspace(0, int(max(self.__y_test)))

        g = sns.FacetGrid(data.get_dataframe(), size = 7)
        g = g.map(plt.scatter, "y_test", "y_predicted", edgecolor='w')
        plt.plot(x_plot, x_plot, color='red', label='Identity line')
        g.set_xlabels('Real values')
        g.set_ylabels('Estimated values')
        plt.legend()
        plt.show()
        
        if savefig is True:
            g.savefig('regression_scatter.png', dpi=600)
        
        if not np.any(self.__y_test<=0) or not np.any(self.__y_pred<=0):
            return {
                'R2': r2_score(self.__y_test, np.squeeze(self.__y_pred)),
                'MSE': mean_squared_error(self.__y_test, np.squeeze(self.__y_pred)),
                'RMSE':sqrt(mean_squared_error(self.__y_test, np.squeeze(self.__y_pred))),
                'MAE': mean_absolute_error(self.__y_test, np.squeeze(self.__y_pred)),
                'MEDAE': median_absolute_error(self.__y_test, np.squeeze(self.__y_pred)),
            }
        return {
                'R2': r2_score(self.__y_test, np.squeeze(self.__y_pred)),
                'MSE': mean_squared_error(self.__y_test, np.squeeze(self.__y_pred)),
                'RMSE':sqrt(mean_squared_error(self.__y_test, np.squeeze(self.__y_pred))),
                'MAE': mean_absolute_error(self.__y_test, np.squeeze(self.__y_pred)),
                'MEDAE': median_absolute_error(self.__y_test, np.squeeze(self.__y_pred)),
                'MSLE': mean_squared_log_error(self.__y_test, np.squeeze(self.__y_pred))
            }
        
    def classification_report(self, y_test=None, y_predicted=None): 
        """
        pass y_test and y_predected as pandas serie is get_column
        """
        if y_test is not None and y_predicted is not None:
            self.__y_test = y_test
        return classification_report(self.__y_test, self.__y_pred)
        
    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.__y_test, self.__y_pred)
        roc_auc = auc(fpr, tpr)
        print("Air sous la courbe" + str(roc_auc)) 
        plt.figure()
        plt.plot(fpr, tpr, color='orange', lw=2, label='ROC curve(area under curve = % 0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='darkgrey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate(1 - Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')
        plt.title('ROC Curve')
        plt.legend(loc='upper left')
        plt.show()

    def boost_model(self):
        ada_boost = AdaBoostClassifier(n_estimators=100, base_estimator=self.__model, learning_rate=0.1, random_state=0)
        self.__boosted_model = ada_boost
        self.__boosted_model.fit(self.__x_train, self.__y_train)

    def predict_with_boosted_model(self, x_to_pred):
        return self.__boosted_model.predict(x_to_pred)

    def save_model(self, model_path='data/model.data'):
        dump(self.__model, model_path) 

    def load_model(self, model_path):
        self.__model = load(model_path)

    def report(self):
        if self.__model_type == 'dl':
            if self.__c_or_r_ts == 'ts' or self.__c_or_r_ts == 'r':
                if self.__validation_percentage == 0:
                    loss = self.history.history['loss']
                    x = range(1, len(loss) + 1)
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(x, loss, 'b', label='Training loss')
                    plt.title('Training and validation loss')
                    plt.legend()
                    plt.show()
                else:
                    loss = self.history.history['loss']
                    val_loss = self.history.history['val_loss']
                    r2 = self.history.history['r2_keras']
                    val_r2 = self.history.history['val_r2_keras']
                    x = range(1, len(loss) + 1)
                    # (1,2) one row and 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.suptitle('Training and validation monitoring of loss and R2')
                    ax1.plot(x, loss, 'b', label='Training loss')
                    ax1.plot(x, val_loss, 'r', label='Validation loss')
                    ax1.set_title('Loss monitoring')
                    ax1.legend()
                    ax2.plot(x, r2, 'b', label='Training R2')
                    ax2.plot(x, val_r2, 'r', label='Validation R2')
                    ax2.set_title('R2 monitoring')
                    ax2.legend()
                print(self.regression_report()) 
                plt.show()
                
            elif self.__c_or_r_ts == 'c':
                if self.__validation_percentage == 0:
                    acc = self.history.history['accuracy']
                    loss = self.history.history['loss']
                    x = range(1, len(acc) + 1)
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    plt.title('Accuracy ')
                    plt.plot(x, acc, 'r', label='Accuracy')
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(x, loss, 'b', label='Loss')
                    plt.title('Loss')
                    plt.legend()
                else:
                    acc = self.history.history['accuracy']
                    val_acc = self.history.history['val_accuracy']
                    loss = self.history.history['loss']
                    val_loss = self.history.history['val_loss']
                    x = range(1, len(acc) + 1)
                    plt.figure(figsize=(12, 5))
                    plt.subplot(2, 2, 1)
                    plt.plot(x, acc, 'b', label='Training accuracy')
                    plt.plot(x, val_acc, 'r', label='Validation accuracy')
                    plt.title('Training and validation accuracy')
                    plt.legend()
                    plt.subplot(2, 2, 2)
                    plt.plot(x, loss, 'b', label='Training loss')
                    plt.plot(x, val_loss, 'r', label='Validation loss')
                    plt.title('Training and validation loss')
                    plt.legend()
        else:
            if self.__c_or_r_ts == 'r':
                print(self.regression_report()) 
            else:
                print(self.classification_report()) 
                
    def cross_validation(self, k):
        """https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter"""
        # scoring = "neg_mean_squared_error"
        if self.__c_or_r_ts == 'r':
            scoring = "r2" 
        elif self.__c_or_r_ts == 'c':
            scoring = "accuracy"
        print(cross_val_score(self.__model, self.x, self.y, cv=k, scoring=scoring))
        
    def get_features_importance(self):
        if self.__c_or_r_ts == 'r':
            etr_model = ExtraTreesRegressor()
            etr_model.fit(self.x,self.y)
            feature_imp = pd.Series(etr_model.feature_importances_,index=self.x.columns)
            feature_imp.nlargest(10).plot(kind='barh')
            plt.show()
        else:
            etr_model = ExtraTreesClassifier()
            etr_model.fit(self.x,self.y)
            feature_imp = pd.Series(etr_model.feature_importances_,index=[i for i in range(self.x.shape[1])])
            feature_imp.nlargest(10).plot(kind='barh')
            plt.show()
            """model = self.__model # or XGBRegressor
            plot_importance(model, importance_type = 'gain') # other options available
            plt.show()
            # if you need a dictionary 
            model.get_booster().get_score(importance_type = 'gain')"""
        return etr_model.feature_importances_
    
    def dt_text_representation(self):
        return tree.export_text(self.__model)
    
    def plot_dt_representation(self, viz_type='graph_viz'):
        if viz_type == 'graph_viz':
            
            # DOT data
            dot_data = tree.export_graphviz(self.__model, out_file=None, 
                                            feature_names=self.x.columns.values,  
                                            class_names=self.y.name,
                                            filled=True)

            # Draw graph
            graph = graphviz.Source(dot_data, format="png") 
            return graph
            
        elif viz_type == 'matplotlib':
            fig = plt.figure(figsize=(25,20))
            _ = tree.plot_tree(self.__model, 
                            feature_names=self.x.columns.values,  
                            class_names=self.y.name,
                            filled=True)
            fig.savefig("decistion_tree.png")
            plt.show()
            
    def viz_reporter(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__y_test, linewidth=3, label='ground truth')
        plt.plot(self.__y_pred, linewidth=3, label='predicted')
        plt.legend(loc='best')
        plt.xlabel('X')
        plt.ylabel('target value')
        
    @staticmethod
    def r2_keras(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred ))
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
