import pandas                as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers                import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection     import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models    import load_model
import numpy  as np

PATH     = "/Users/user/Desktop/BCIT/Machine Learning/assignment1/"
CSV_DATA = "Walmart_Store_sales_light.csv"
df  = pd.read_csv(PATH + CSV_DATA)

def addShiftedColumn(df):
    finalDF = pd.DataFrame()
    numberOfStores = set(df['Store'])
    for num in numberOfStores:
        tempDF = df[df['Store'] == num]
        tempDF['Date'] = pd.to_datetime(tempDF['Date'])
        tempDF = tempDF.sort_values(by=['Date'], ascending=False)
        tempDF["Weekly_Sales_t-1"] = tempDF["Weekly_Sales"].shift(1)
        tempDF = tempDF.dropna().reset_index(drop=True)
        finalDF = finalDF.append(tempDF)
    return finalDF

df = addShiftedColumn(df)

# Binning
df['CPI'] = np.digitize(df['CPI'], [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

columnsX = ['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', "Weekly_Sales_t-1"]
columnsY = ['Weekly_Sales']

X = pd.DataFrame(MinMaxScaler().fit_transform(df[columnsX]), columns=columnsX)
y = pd.DataFrame(MinMaxScaler().fit_transform(df[columnsY]), columns=columnsY)
# y = df[columnsY]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the model.
def create_model(nodes, learning_rate, momentum, activation, kernel_initializer, number_of_layers):
    model = Sequential()
    for _ in range(number_of_layers):
        model.add(Dense(nodes, activation=activation,kernel_initializer=kernel_initializer))
    model.add(Dense(1, activation=activation))
    opt = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='mse', metrics=['accuracy', 'mse'], optimizer=opt)

    # simple early stopping
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1, callbacks=[mc])

    return model

param_grid = {
    'nodes': [100],
    'learning_rate': [0.01],
    'momentum': [0.9],
    'number_of_layers': [1],
    'activation': ['relu'],
    'kernel_initializer': ['he_uniform'],  
}

# param_grid = {
#     'nodes': [25, 50, 100],
#     'learning_rate': [0.01, 0.001],
#     'momentum': [0.9, 1],
#     'number_of_layers': [1, 2, 3],
#     'activation': ['relu', 'softplus', 'softmax'],
#     'kernel_initializer': ['he_uniform', 'uniform', 'normal'],  
# }

model = KerasRegressor(build_fn=create_model)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1)
grid_result = grid.fit(X_train, y_train)

# summarize results
means   = grid_result.cv_results_['mean_test_score']
stds    = grid_result.cv_results_['std_test_score']
params  = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print(" **** BEST ****: %f using %s" % (grid_result.best_score_, grid_result.best_params_))