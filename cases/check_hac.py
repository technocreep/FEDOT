import os
import timeit
import pandas as pd
import numpy as np
from typing import List

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from fedot.core.utils import fedot_project_root


def run_exp(clip_data: bool = False,  safe_mode: bool = False):
    """ Start classification task with desired dataset
    :param clip_data: is there a need to clip dataframe
    :param safe_mode: is there a need to perform safe mode in AutoML
    """

    data = pd.read_csv('raif.csv')
    print(f'Dataset size: {data.shape}')
    if clip_data is True:
        # Crop dataset for albert correctness check
        data = data.iloc[:5000]

    column_names = np.array(data.columns)
    id_target = np.ravel(np.argwhere(column_names == 'per_square_meter_price'))
    column_names = list(column_names)
    column_names.pop(id_target[0])

    predictors = np.array(data[column_names])
    target = np.array(data['per_square_meter_price'])

    start = timeit.default_timer()
    fedot = Fedot(problem='classification', timeout=0.5, safe_mode=safe_mode)
    pipeline = fedot.fit(features=predictors, target=target, predefined_model='dt')
    fedot_predict = np.ravel(fedot.predict(predictors))
    print(fedot_predict)
    time_launch = timeit.default_timer() - start
    print(f'Fitting takes {time_launch:.2f} seconds')


if __name__ == '__main__':
    run_exp(clip_data=False, safe_mode=True)