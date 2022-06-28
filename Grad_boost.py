from sklearn.preprocessing import MultiLabelBinarizer
from catboost import CatBoostRegressor
import pandas as pd

from numpy import ndarray


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    cats = ['genres', 'directors', 'filming_locations', 'actor_0_gender', 'actor_1_gender', 'actor_2_gender']
    df_train.pop('keywords')
    df_test.pop('keywords')
    for i, cat in enumerate(cats[:3]):
        df_train[cat] = df_train[cat].apply(lambda x: [x + str(i)] if x == 'unknown' else x)
        df_test[cat] = df_test[cat].apply(lambda x: [x + str(i)] if x == 'unknown' else x)

    for cat in cats[3:]:
        df_train[cat] = df_train[cat].apply(lambda x: [x])
        df_test[cat] = df_test[cat].apply(lambda x: [x])

    dic = {}
    for cat in cats[:3]:
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(df_train.pop(cat))
        dic[cat] = mlb
        df_train = df_train.join(pd.DataFrame(arr, index=df_train.index, columns=mlb.classes_))
    for i, cat in enumerate(cats[3:]):
        mlb = MultiLabelBinarizer()
        arr = mlb.fit_transform(df_train.pop(cat))
        dic[cat] = mlb
        classes = [x +str(i) for x in mlb.classes_]
        df_train = df_train.join(pd.DataFrame(arr, index=df_train.index, columns=classes))

    y = df_train.pop('awards')
    X = df_train

    for cat in cats[:3]:
        arr = dic[cat].transform(df_test.pop(cat))
        df_test = df_test.join(pd.DataFrame(arr, index=df_test.index, columns=dic[cat].classes_))
    for i, cat in enumerate(cats[3:]):
        arr = dic[cat].transform(df_test.pop(cat))
        classes = [x +str(i) for x in dic[cat].classes_]
        df_test = df_test.join(pd.DataFrame(arr, index=df_test.index, columns=classes))


    X_test = df_test
    params = {
        'learning_rate' : 0.049570101661787496,
        'n_estimators' : 800
    }
    regressor = CatBoostRegressor(train_dir='/tmp/catboost_info', **params, verbose=0)
    regressor.fit(X, y)
    return regressor.predict(X_test)