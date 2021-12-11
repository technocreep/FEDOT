# Additional imports are required
import os

import numpy as np
import pandas as pd

from fedot.core.data.multi_modal import MultiModalData
from sklearn.metrics import mean_absolute_error

# Plots
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 18, 7

import warnings
warnings.filterwarnings('ignore')

# Prerocessing for FEDOT
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

# FEDOT
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode


def plot_results(actual_time_series, predicted_values, len_train_data,
                 y_name='Parameter'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


df = pd.read_csv('ts_sea_level.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Specify forecast length
len_forecast = 250
task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

# Got train, test parts, and the entire data
true_values = np.array(df['Level'])
train_array = true_values[:-len_forecast]
test_array = true_values[-len_forecast:]

# Data for lagged transformation
train_lagged = InputData(idx=np.arange(0, len(train_array)),
                         features=train_array,
                         target=train_array,
                         task=task,
                         data_type=DataTypesEnum.ts)
start_forecast = len(train_array)
end_forecast = start_forecast + len_forecast
predict_lagged = InputData(idx=np.arange(start_forecast, end_forecast),
                           features=train_array,
                           target=test_array,
                           task=task,
                           data_type=DataTypesEnum.ts)

exog_arr = np.array(df['Neighboring level'])
exog_train = exog_arr[:-len_forecast]
exog_test = exog_arr[-len_forecast:]

# Data for exog operation
train_exog = InputData(idx=np.arange(0, len(exog_train)),
                       features=exog_train,
                       target=train_array,
                       task=task,
                       data_type=DataTypesEnum.ts)
start_forecast = len(exog_train)
end_forecast = start_forecast + len_forecast
predict_exog = InputData(idx=np.arange(start_forecast, end_forecast),
                          features=exog_test,
                          target=test_array,
                          task=task,
                          data_type=DataTypesEnum.ts)

train_dataset = MultiModalData({
    'lagged': train_lagged,
    'exog_ts': train_exog
})

predict_dataset = MultiModalData({
    'lagged': predict_lagged,
    'exog_ts': predict_exog
})

# Create a pipeline with different data sources in th nodes
node_lagged = PrimaryNode('lagged')
node_exog = PrimaryNode('exog_ts')
node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged, node_exog])
exog_pipeline = Pipeline(node_ridge)

# Fit it
exog_pipeline.fit(train_dataset)

# Predict
predicted = exog_pipeline.predict(predict_dataset)
predicted_values = np.ravel(np.array(predicted.predict))

# Plot predictions and true values
plot_results(actual_time_series = true_values,
             predicted_values = predicted_values,
             len_train_data = len(train_array),
             y_name = 'Sea level, m')

# Print MAE metric
print(f'Mean absolute error: {mean_absolute_error(test_array, predicted_values):.3f}')