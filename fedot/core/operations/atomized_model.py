from typing import Callable, Union, Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.model import assign_model_tabular_column_types
from fedot.core.operations.operation import Operation
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationMetaInfo, \
    atomized_model_type
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.utils import make_pipeline_generator


class AtomizedModel(Operation):
    """ Class which replace Operation class for AtomizedModel object """

    def __init__(self, pipeline: 'Pipeline'):
        if not pipeline.root_node:
            raise ValueError(f'AtomizedModel could not create instance of empty Pipeline.')

        super().__init__(operation_type=atomized_model_type())
        self.pipeline = pipeline
        self.unique_id = self.pipeline.root_node.descriptive_id

    def fit(self, params: Optional[Union[str, dict]], data: InputData,
            is_fit_pipeline_stage: bool = True,
            use_cache: bool = True):

        predicted_train = self.pipeline.fit(input_data=data)
        fitted_atomized_operation = self.pipeline

        return fitted_atomized_operation, predicted_train

    def predict(self, fitted_operation, data: InputData, is_fit_pipeline_stage: bool = False,
                params: Optional[Union[str, dict]] = None, output_mode: str = 'default'):
        prediction = fitted_operation.predict(input_data=data, output_mode=output_mode)
        prediction = self.assign_tabular_column_types(prediction, output_mode)
        return prediction

    def fine_tune(self, loss_function: Callable,
                  loss_params: dict = None,
                  input_data: InputData = None, iterations: int = 50,
                  timeout: int = 5):
        """ Method for tuning hyperparameters """
        tuned_pipeline = self.pipeline.fine_tune_all_nodes(loss_function=loss_function,
                                                           loss_params=loss_params,
                                                           input_data=input_data,
                                                           iterations=iterations,
                                                           timeout=timeout)
        tuned_atomized_model = AtomizedModel(tuned_pipeline)
        return tuned_atomized_model

    @property
    def metadata(self) -> OperationMetaInfo:
        generator = make_pipeline_generator(self.pipeline)
        tags = set()

        for node in generator:
            tags.update(node.operation_tags)

        root_node = self.pipeline.root_node
        supported_strategies = None
        allowed_positions = ['any']
        tags = list(tags)

        operation_info = OperationMetaInfo(root_node.operation.supplementary_data.id,
                                           root_node.operation.supplementary_data.input_types,
                                           root_node.operation.supplementary_data.output_types,
                                           root_node.operation.supplementary_data.task_type,
                                           supported_strategies, allowed_positions,
                                           tags)
        return operation_info

    def description(self, operation_params: Optional[dict]):
        operation_type = self.operation_type
        operation_length = self.pipeline.length
        operation_depth = self.pipeline.depth
        operation_id = self.unique_id
        operation_types = {}

        for node in self.pipeline.nodes:
            if node.operation.operation_type in operation_types:
                operation_types[node.operation.operation_type] += 1
            else:
                operation_types[node.operation.operation_type] = 1

        return f'{operation_type}_length:{operation_length}_depth:{operation_depth}' \
               f'_types:{operation_types}_id:{operation_id}'

    @staticmethod
    def assign_tabular_column_types(output_data: OutputData,
                                    output_mode: str) -> OutputData:
        """
        Assign types for tabular data obtained from model predictions.
        By default, all types of model predictions for tabular data can be clearly defined
        """
        return assign_model_tabular_column_types(output_data, output_mode)
