"""
Under MIT License by EL HACHIMI CHOUAIB
"""
import tensorflow.keras.losses
import sklearn.tree as tree
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool1D, Conv1D, Reshape, LSTM
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error, mean_absolute_error, classification_report, precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB


class Model:
    def __init__(
        self, 
        data_x=None, 
        data_y=None, 
        model_type='knn', 
        c_or_r_or_ts='c',
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
        self.__c_or_r_ts = c_or_r_or_ts
        self.__validation_percentage = validation_percentage
        
        if model_type == 'dt':
            if c_or_r_or_ts == 'c':
                self.__model = tree.DecisionTreeClassifier()
            else:
                self.__model = tree.DecisionTreeRegressor()

        elif model_type == 'svm':
            if c_or_r_or_ts == 'c':
                self.__model = svm.SVC()
            else:
                self.__model = svm.SVR()
                
        elif model_type == 'lr':
            if c_or_r_or_ts == 'c':
                self.__model = LogisticRegression(random_state=2)
            else:
                self.__model = LinearRegression()

        elif model_type == 'nb':
            if c_or_r_or_ts == 'c':
                self.__model = MultinomialNB()
            else:
                self.__model = GaussianNB()
        
        elif model_type == 'rf':
            if c_or_r_or_ts == 'c':
                self.__model = RandomForestClassifier()
            else:
                self.__model = RandomForestRegressor()
                
        elif model_type == 'xb':
            if c_or_r_or_ts == 'c':
                self.__model = XGBClassifier()
            else:
                self.__model = XGBRegressor()

        elif model_type == 'dl':
            self.__model = Sequential()

        elif model_type == 'knn':
            if c_or_r_or_ts == 'c':
                self.__model = KNeighborsClassifier(n_neighbors=5)
            else:
                self.__model = KNeighborsRegressor(n_neighbors=5)
            
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
        if input_dim:
            self.__model.add(Dense(connections_number, activation=activation_function, input_dim=input_dim))
        else:
            self.__model.add(Dense(connections_number, activation=activation_function))
            
    def add_lstm_layer(self, connections_number=2, activation_function='relu', input_shape=None):
        if input_shape is not None:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, input_shape=input_shape, return_sequences=True))
        else:
            self.__model.add(LSTM(units=connections_number, activation=activation_function, return_sequences=True))

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

    def train(self, loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.SGD(lr=0.001), metrics_as_list=['accuracy']):
        """
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
            self.__model.compile(loss=loss, optimizer=optimizer, metrics=[tensorflow.keras.metrics.RootMeanSquaredError()])
            if self.__generator is not None:
                self.history = self.__model.fit(self.get_generator(), epochs=self.__epochs, batch_size=self.__batch_size)
                print(self.history.history)
            else:
                self.history = self.__model.fit(self.x, self.y, epochs=self.__epochs,
                                        batch_size=self.__batch_size, validation_split=self.__validation_percentage)
                print(self.history.history)
        else:
            self.__model.fit(self.__x_train, self.__y_train)
            self.__y_pred = self.__model.predict(self.__x_test)

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

    def regression_report(self, y_test=None, y_predicted=None): 
        """
        pass y_test and y_predected as pandas serie is get_column
        """
        
        if y_test is not None and y_predicted is not None:
            self.__y_test = y_test
            self.__y_pred = y_predicted
        return {
            'R2': r2_score(self.__y_test, self.__y_pred),
            'MSE': mean_squared_error(self.__y_test, self.__y_pred),
            'RMSE':sqrt(mean_squared_error(self.__y_test, self.__y_pred)),
            'MAE': mean_absolute_error(self.__y_test, self.__y_pred),
            'MEDAE': median_absolute_error(self.__y_test, self.__y_pred),
            'MSLE': mean_squared_log_error(self.__y_test, self.__y_pred)
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
                    x = range(1, len(loss) + 1)
                    plt.subplot(1, 2, 2)
                    plt.plot(x, loss, 'b', label='Training loss')
                    plt.plot(x, val_loss, 'r', label='Validation loss')
                    plt.title('Training and validation loss')
                    plt.legend()
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
                    plt.subplot(1, 2, 1)
                    plt.plot(x, acc, 'b', label='Training accuracy')
                    plt.plot(x, val_acc, 'r', label='Validation accuracy')
                    plt.title('Training and validation accuracy')
                    plt.legend()
                    plt.subplot(1, 2, 2)
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
            feature_imp = pd.Series(etr_model.feature_importances_,index=self.x.columns)
            feature_imp.nlargest(10).plot(kind='barh')
            plt.show()
        return etr_model.feature_importances_