from typing import Union
import numpy as np
import pandas as pd
from fedot.api.api_utils.data_definition import data_strategy_selector
from fedot.core.log import Log
from fedot.preprocessing.preprocessing import DataPreprocessor
from fedot.core.utils import probs_to_labels
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.pipelines.pipeline import Pipeline


class ApiDataProcessor:
    """
    Class for selecting optimal data processing strategies based on type of data.
    Available data sources are:
        * numpy array
        * pandas DataFrame
        * string (path to csv file)
        * InputData (FEDOT dataclass)

    Data preprocessing such a class performing also
    """

    def __init__(self, task: Task, log: Log = None):
        self.task = task
        self.preprocessor = DataPreprocessor(log)

    def define_data(self,
                    features: Union[str, np.ndarray, pd.DataFrame, InputData, dict],
                    target: Union[str, np.ndarray, pd.Series] = None,
                    is_predict=False):
        """ Prepare data for fedot pipeline composing.
        Obligatory preprocessing steps are applying also
        """
        try:
            data = data_strategy_selector(features=features,
                                          target=target,
                                          ml_task=self.task,
                                          is_predict=is_predict)
        except Exception as ex:
            raise ValueError('Please specify a features as path to csv file, as Numpy array, '
                             'Pandas DataFrame, FEDOT InputData or dict for multimodal data')

        # Perform obligatory steps of data preprocessing
        if is_predict is False:
            data = self.preprocessor.obligatory_prepare_for_fit(data)
        else:
            data = self.preprocessor.obligatory_prepare_for_predict(data)
        data.supplementary_data.was_preprocessed = True
        return data

    def define_predictions(self, current_pipeline: Pipeline, test_data: InputData):

        if self.task.task_type == TaskTypesEnum.classification:
            prediction = current_pipeline.predict(test_data, output_mode='labels')
            output_prediction = prediction
        elif self.task.task_type == TaskTypesEnum.ts_forecasting:
            # Convert forecast into one-dimensional array
            prediction = current_pipeline.predict(test_data)
            forecast = np.ravel(np.array(prediction.predict))
            prediction.predict = forecast
            output_prediction = prediction
        else:
            prediction = current_pipeline.predict(test_data)
            output_prediction = prediction

        return output_prediction

    def correct_shape(self, metric_name: str,
                      real: InputData, prediction: OutputData):
        """ Change shape for models predictions if its necessary """
        if self.task == TaskTypesEnum.ts_forecasting:
            real.target = real.target[~np.isnan(prediction.predict)]
            prediction.predict = prediction.predict[~np.isnan(prediction.predict)]

        if metric_name == 'f1':
            if len(prediction.predict.shape) > len(real.target.shape):
                prediction.predict = probs_to_labels(prediction.predict)
            elif real.num_classes == 2:
                prediction.predict = probs_to_labels(self.convert_to_two_classes(prediction.predict))
        return real.target, prediction.predict

    @staticmethod
    def convert_to_two_classes(predict):
        return np.vstack([1 - predict, predict]).transpose()
