import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TaskParams
from sklearn.metrics import classification_report

random.seed(1)
np.random.seed(1)

NPY_SAVE_PATH = os.path.join(os.path.abspath(os.path.curdir), 'npy_data')


def get_initial_pipeline():
    """ Return initial pipeline """

    ds_regr = PrimaryNode('data_source_table/regr')
    ds_class = PrimaryNode('data_source_table/class')

    scaling_node_regr = SecondaryNode('scaling', nodes_from=[ds_regr])
    scaling_node_class = SecondaryNode('scaling', nodes_from=[ds_class])

    pca_node_regr = SecondaryNode('pca', nodes_from=[scaling_node_regr])
    pca_node_regr.custom_params = {'n_components': 0.2}

    pca_node_class = SecondaryNode('pca', nodes_from=[scaling_node_class])
    pca_node_class.custom_params = {'n_components': 0.2}

    # imp_regr = SecondaryNode('simple_imputation', nodes_from=[ds_regr])
    # imp_class = SecondaryNode('simple_imputation', nodes_from=[ds_class])

    regr_node = SecondaryNode('ridge', nodes_from=[scaling_node_regr])
    class_node = SecondaryNode('logit', nodes_from=[scaling_node_class])

    root = SecondaryNode('dt', nodes_from=[regr_node, class_node])

    initial_pipeline = Pipeline(root)

    initial_pipeline.show()

    return initial_pipeline


def save_predictions(x_test, y_test, predict):
    # Save predictions in csv file
    result_dict = {}
    n_rows, n_cols = x_test.shape
    for i in range(n_cols):
        result_dict.update({f'feature_{i}': x_test[:, i]})
    result_dict.update({'predict': predict})
    result_dict.update({'actual': y_test})

    df = pd.DataFrame(result_dict)
    df.to_csv('predictions.csv', index=False)


def run_fedot(timeout: float = 0.5, features_for_train: int = None):
    """ Launch fedot on dataset
    :param timeout: time for AutoML in minutes
    :param features_for_train: how many columns (features) to use from dataset
    """
    x_train = np.load(os.path.join(NPY_SAVE_PATH, 'x_train.npy'))
    y_train = np.load(os.path.join(NPY_SAVE_PATH, 'y_train.npy'))

    x_test = np.load(os.path.join(NPY_SAVE_PATH, 'x_test.npy'))
    y_test = np.load(os.path.join(NPY_SAVE_PATH, 'y_test.npy'))

    x_val = np.load(os.path.join(NPY_SAVE_PATH, 'x_val.npy'))
    y_val = np.load(os.path.join(NPY_SAVE_PATH, 'y_val.npy'))

    if features_for_train is not None:
        # Clip arrays
        x_train = x_train[:, :features_for_train]
        x_test = x_test[:, :features_for_train]
        x_val = x_val[:, :features_for_train]

    available_operations = ['bernb', 'dt', 'knn', 'lda', 'qda', 'logit', 'rf', 'svc',
                            'scaling', 'normalization', 'pca', 'kernel_pca']

    initial_pipeline = get_initial_pipeline()
    automl = Fedot(problem='regression', timeout=timeout,
                   verbose_level=4, composer_params={'timeout': timeout,
                                                     'available_operations': available_operations},
                   initial_assumption=initial_pipeline)

    idx = np.arange(len(y_train))
    train_regr = InputData(idx=idx, features=x_train, target=y_train.reshape((-1, 1)), data_type=DataTypesEnum.table,
                           task=Task(TaskTypesEnum.regression,
                                     task_params=TaskParams(is_main_task=True)),
                           supplementary_data=SupplementaryData(is_main_target=True, was_preprocessed=False))

    train_class = InputData(idx=idx, features=x_train, target=y_train.reshape((-1, 1)), data_type=DataTypesEnum.table,
                            task=Task(TaskTypesEnum.classification,
                                      task_params=TaskParams(is_main_task=False)),
                            supplementary_data=SupplementaryData(is_main_target=False, was_preprocessed=False))

    predict_regr = InputData(idx=idx, features=x_test, target=None, data_type=DataTypesEnum.table,
                             task=Task(TaskTypesEnum.regression,
                                       task_params=TaskParams(is_main_task=True)),
                             supplementary_data=SupplementaryData(is_main_target=False, was_preprocessed=False))

    predict_class = InputData(idx=idx, features=x_test, target=None, data_type=DataTypesEnum.table,
                              task=Task(TaskTypesEnum.classification,
                                        task_params=TaskParams(is_main_task=False)),
                              supplementary_data=SupplementaryData(is_main_target=False, was_preprocessed=False))

    fit_data = MultiModalData({
        'data_source_table/regr': train_regr,
        'data_source_table/class': train_class
    })

    predict_data = MultiModalData({
        'data_source_table/regr': predict_regr,
        'data_source_table/class': predict_class
    })

    initial_pipeline.fit(fit_data)

    # pipeline = automl.fit(fit_data, predefined_model=initial_pipeline)
    # predict = automl.predict(predict_data)
    #
    # # Display obtained composite model structure
    # pipeline.show()
    #
    # # Serialize pipeline with name timeout and features_for_train info
    # if features_for_train is None:
    #     features_for_train = 'all'
    # pipeline.save(f'pipeline_{timeout}_{features_for_train}')
    #
    # # Features and predictions into csv file
    # save_predictions(x_test, y_test, predict)
    #
    # # Calculate metrics
    # print(classification_report(y_test, predict))


if __name__ == '__main__':
    run_fedot(timeout=10, features_for_train=None)
