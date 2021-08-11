from keras import Input
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import KFold


class RegressionNetwork(object):
    def __init__(self, input_size, layers_sizes, output_size):
        model = Sequential()
        model.add(Input(input_size))
        for layer_size in layers_sizes:
            model.add(Dense(layer_size, kernel_initializer='normal', activation='relu'))
        model.add(Dense(output_size))
        self.model = model

    def compile_model(self):
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        return self.model

    def train(self, x, y, num_folds=2):
        kfold = KFold(n_splits=num_folds, shuffle=True)
        for train, test in kfold.split(x, y):
            model = self.compile_model()
            history = model.fit(x[train], y[test], epochs=100, batch_size=32, verbose=1, validation_split=0.2)
            # visualize_history(history)
            scores = model.evaluate(x[test], y[test], verbose=1)

            print(scores)







