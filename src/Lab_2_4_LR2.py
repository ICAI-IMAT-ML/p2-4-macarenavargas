import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # X sera algo como:
        # X = np.array([1,2,3],[1,4,5],[1,8,9],[1,4,3]) donde la columna de unos es el bias
        # y las otras dos columnas son los features x1 y x2

        # y sera algo como y= np.array([1,2,3,4,5])

        X_t=np.transpose(X)
        X_tX=np.dot(X_t,X)
        w=np.dot(np.dot(np.linalg.inv(X_tX),X_t),y)
        b=w[0]
        w=w[1:]
        self.intercept = b
        self.coefficients = w

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        # tus observaciones qu quieres fit tienen la pinta (x_i,y_i)
        m = len(y)
        self.coefficients = (
            np.random.rand(X.shape[1] - 1) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        

        # Implement gradient descent (TODO)
        for epoch in range(iterations):
            
            predictions = self.predict(X[:,1:])
            error = predictions - y
            # para w
           
            gradient= (2/m) * np.dot(error, X)
         
          
            self.intercept -= learning_rate* gradient[:1]
            self.coefficients -= learning_rate* gradient[1:]

            # TODO: Calculate and print the loss every 10 epochs
            if epoch % 1000 == 0:
                mse = 1/len(y)*sum(error**2)
                print(f"Epoch {epoch}: MSE = {mse}")
    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        # X son nuevos datos de entrada
        # Si X es multidmensional sera ej:
        # X = np.array([[1, 2], [3, 4], [5, 6]]) 3 observaciones, 2 features x1 x2
        # si X es unidimensional sera:
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)
               
        # Multiplicamos nuestra matriz X por la lista de coefcientes
        # Ej, self.coeficientes= np.array([2.5,3]) se multiplicaria cada
        # lista de X por su elemento correspondiente
        # como X tiene el bias, hacemos un slice
        predictions = np.dot(X,self.coefficients) +self.intercept

        # necesitamos el flatten porque en el caso de una dimension necesitamos volver aconvertirlo en un array de una lista
    
        return predictions.flatten()


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    
    RSS= np.sum((y_true-y_pred)**2)
    TSS= np.sum((y_true-np.mean(y_true))**2)
    r_squared = 1- RSS/TSS

    # Root Mean Squared Error

    n=y_pred.shape
    rmse = np.sqrt(np.sum(((y_pred-y_true)**2)/n))

    # Mean Absolute Error

    mae = np.sum(abs(y_true-y_pred)/n)

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()
    
    for index in sorted(categorical_indices, reverse=True):
        # TODO: Extract the categorical column
        categorical_column = X[:, index]
      
        # TODO: Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        # TODO: Create a one-hot encoded matrix (np.array) for the current categorical column
        matrix=[]
        for value in unique_values:
           
            column=[]
            for value_instance in categorical_column:
                if value==value_instance:
                    
                    column.append(1)
                else:
                    column.append(0)
            matrix.append(column)
      
      

        one_hot = np.array(matrix).T
        
        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        # TODO: Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_transformed=np.delete(X_transformed,index,axis=1)
     
        X_transformed=np.concatenate((one_hot,X_transformed), axis=1)

        

    return X_transformed