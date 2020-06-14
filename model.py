from sklearn.linear_model import LinearRegression
import pandas as pd

class rollingModel:
    """
    This class is to implement rolling predictions given a sklearn model. IT
    """
    def __init__(self, period, inner_model=LinearRegression, inner_model_inputs={}):
        self.model_class = inner_model
        self.model = inner_model(**inner_model_inputs)
        self.model_inputs = inner_model_inputs
        self.period=period
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def rolling_predict(self, X_test, y_test):
        out = list()
        for i in range(0,len(X_test),self.period):
            now_X = X_test.iloc[i:i+self.period]
            now_y = y_test.iloc[i:i+self.period]
            now_pred = self.predict(now_X)
            out += list(now_pred)
            self.X_train = pd.concat([self.X_train, now_X])
            self.y_train = pd.concat([self.y_train, now_y])
            self.model = self.model_class(**self.model_inputs)
            self.model.fit(self.X_train, self.y_train)
        return out


if __name__ == '__main__':
    import pandas as pd
    import pickle

    X = pd.read_csv('data/engineered_features.csv',index_col=0)
    y = pickle.load(open('data/target.pkl', 'rb'))

    index_till_train = int(len(X) * 2 / 3)
    X_train, y_train = X.iloc[:index_till_train], y.iloc[:index_till_train][0]
    X_test, y_test = X.iloc[index_till_train:], y.iloc[index_till_train:][0]

    model = rollingModel(period=5)
    model.fit(X_train, y_train)
    out = model.rolling_predict(X_test, y_test)


