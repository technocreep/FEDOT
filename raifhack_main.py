import typing
from enum import IntEnum

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

from fedot.api.main import Fedot
from fedot.core.utils import fedot_project_root

UNKNOWN_VALUE = 'missing'

# Categorical
CATEGORICAL_STE_FEATURES = ['region', 'city', 'realty_type']

# One-hot encoded features
CATEGORICAL_OHE_FEATURES = []

# Numerical
NUM_FEATURES = [
    'lat',
    'lng',
    'osm_amenity_points_in_0.001',
    'osm_amenity_points_in_0.005',
    'osm_amenity_points_in_0.0075',
    'osm_amenity_points_in_0.01',
    'osm_building_points_in_0.001',
    'osm_building_points_in_0.005',
    'osm_building_points_in_0.0075',
    'osm_building_points_in_0.01',
    'osm_catering_points_in_0.001',
    'osm_catering_points_in_0.005',
    'osm_catering_points_in_0.0075',
    'osm_catering_points_in_0.01',
    'osm_city_closest_dist',
    'osm_city_nearest_population',
    'osm_crossing_closest_dist',
    'osm_crossing_points_in_0.001',
    'osm_crossing_points_in_0.005',
    'osm_crossing_points_in_0.0075',
    'osm_crossing_points_in_0.01',
    'osm_culture_points_in_0.001',
    'osm_culture_points_in_0.005',
    'osm_culture_points_in_0.0075',
    'osm_culture_points_in_0.01',
    'osm_finance_points_in_0.001',
    'osm_finance_points_in_0.005',
    'osm_finance_points_in_0.0075',
    'osm_finance_points_in_0.01',
    'osm_healthcare_points_in_0.005',
    'osm_healthcare_points_in_0.0075',
    'osm_healthcare_points_in_0.01',
    'osm_historic_points_in_0.005',
    'osm_historic_points_in_0.0075',
    'osm_historic_points_in_0.01',
    'osm_hotels_points_in_0.005',
    'osm_hotels_points_in_0.0075',
    'osm_hotels_points_in_0.01',
    'osm_leisure_points_in_0.005',
    'osm_leisure_points_in_0.0075',
    'osm_leisure_points_in_0.01',
    'osm_offices_points_in_0.001',
    'osm_offices_points_in_0.005',
    'osm_offices_points_in_0.0075',
    'osm_offices_points_in_0.01',
    'osm_shops_points_in_0.001',
    'osm_shops_points_in_0.005',
    'osm_shops_points_in_0.0075',
    'osm_shops_points_in_0.01',
    'osm_subway_closest_dist',
    'osm_train_stop_closest_dist',
    'osm_train_stop_points_in_0.005',
    'osm_train_stop_points_in_0.0075',
    'osm_train_stop_points_in_0.01',
    'osm_transport_stop_closest_dist',
    'osm_transport_stop_points_in_0.005',
    'osm_transport_stop_points_in_0.0075',
    'osm_transport_stop_points_in_0.01',
    'reform_count_of_houses_1000',
    'reform_count_of_houses_500',
    'reform_house_population_1000',
    'reform_house_population_500',
    'reform_mean_floor_count_1000',
    'reform_mean_floor_count_500',
    'reform_mean_year_building_1000',
    'reform_mean_year_building_500',
    'total_square'
]

# Target
TARGET = 'per_square_meter_price'


class PriceTypeEnum(IntEnum):
    OFFER_PRICE = 0  # цена из объявления
    MANUAL_PRICE = 1  # цена, полученная путем ручной оценки


THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1
EPS = 1e-8


def deviation_metric_one_sample(y_true: typing.Union[float, int], y_pred: typing.Union[float, int]) -> float:
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9


def deviation_metric(y_true: np.array, y_pred: np.array) -> float:
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()


def median_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.median(np.abs(y_pred - y_true) / y_true)


def metrics_stat(y_true: np.array, y_pred: np.array) -> typing.Dict[str, float]:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mdape = median_absolute_percentage_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    raif_metric = deviation_metric(y_true, y_pred)
    return {'mape': mape, 'mdape': mdape, 'rmse': rmse, 'r2': r2, 'raif_metric': raif_metric}


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет пропущенные категориальные переменные
    :param df: dataframe, обучающая выборка
    :return: dataframe
    """
    df_new = df.copy()
    fillna_cols = ['region', 'city', 'street', 'realty_type']
    df_new[fillna_cols] = df_new[fillna_cols].fillna(UNKNOWN_VALUE)
    return df_new


def predict(m, x, y, xval):
    predictions = m.predict(features=x)
    deviation = ((y - predictions) / predictions).median()
    predictions = m.predict(xval)
    corrected_price = predictions * (1 + deviation)
    return corrected_price


def measure(m, X_offer, y_offer, X_offer_val, y_offer_val, X_manual, y_manual, X_manual_val, y_manual_val):
    print('TRAIN_RESULTS\nOFFER:')
    print(*metrics_stat(y_offer.values, m.predict(X_offer)).items(), sep='\n')
    print('MANUAL: ')
    print(*metrics_stat(y_manual.values, predict(m, X_manual, y_manual, X_manual)).items(), sep='\n')
    print('VAL_RESULTS\nOFFER:')
    print(*metrics_stat(y_offer_val.values, m.predict(X_offer_val)).items(), sep='\n')
    print('MANUAL: ')
    print(*metrics_stat(y_manual_val.values, predict(m, X_manual, y_manual, X_manual_val)).items(), sep='\n')


if __name__ == '__main__':
    # Data preparation
    train_data_path = f'{fedot_project_root()}/cases/data/rh_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/rh_test.csv'
    submission_path = f'{fedot_project_root()}/cases/data/rh_test_submission.csv'

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    submission = pd.read_csv(submission_path)

    # Validation
    # If year > 2020, than 100% of data goes to train
    # Else data before specified year goes to validation data
    mask = (train_data.date < '2029-07-19')
    train_df, val_df = train_data[mask], train_data[~mask]
    print('train_df.shape:', train_df.shape, 'val_df.shape:', val_df.shape)

    # Prepare categorical
    # train_df = prepare_categorical(train_df)
    # val_df = prepare_categorical(val_df)

    # Dataset separation
    X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
    y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]

    X_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
    y_manual = train_df[train_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]

    X_offer_val = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
    y_offer_val = val_df[val_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]

    X_manual_val = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
    y_manual_val = val_df[val_df.price_type == PriceTypeEnum.MANUAL_PRICE][TARGET]

    # Prepare FEDOT
    problem = 'regression'

    model = Fedot(problem=problem, seed=1518,
                  composer_params={'metric': ['mape'], 'pop_size': 6, 'with_tuning': False},
                  verbose_level=3, timeout=1)
    pipeline = model.fit(features=X_offer, target=y_offer)
    pipeline.show()
    prediction = model.predict(features=test_data)

    print(model.get_metrics())
    model.best_models.show()

    # Look at metrics
    measure(model, X_offer, y_offer, X_offer_val, y_offer_val, X_manual, y_manual, X_manual_val, y_manual_val)

    # Prediction
    test_df = prepare_categorical(test_data)
    test_df['per_square_meter_price'] = predict(model, X_manual, y_manual, test_df[
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES])

    # Submission prepare
    test_df[['id', 'per_square_meter_price']].to_csv('submission.csv', index=False)
