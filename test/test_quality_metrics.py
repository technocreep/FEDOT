import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import Data, normalize, split_train_test
from core.models.evaluation import EvaluationStrategy
from core.models.model import LogRegression
from core.repository.quality_metrics_repository import MetricsRepository, ComplexityMetricsEnum, \
    ClassificationMetricsEnum


@pytest.fixture()
def data_setup():
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = normalize(predictors[:100])
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = Data(features=train_data_x, target=train_data_y,
                      idx=np.arange(0, len(train_data_y)))
    test_data = Data(features=test_data_x, target=test_data_y,
                     idx=np.arange(0, len(test_data_y)))
    data = Data(features=predictors, target=response, idx=np.arange(0, 100))
    return train_data, test_data, data


def test_structural_quality(data_setup):
    _, _, data = data_setup

    metric_functions = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)

    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)

    metric_value = metric_functions(chain, data)
    assert metric_value == 13


def test_classification_quality_metric(data_setup):
    _, _, data = data_setup

    metric_functions = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)

    metric_value = metric_functions(chain, data)
    assert metric_value > 0
    assert metric_value < 1
