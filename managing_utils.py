import pandas as pd
import numpy as np
import json


def one_hot_encode(df, colNames):
    for col in colNames:
        if(df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


def simple_encode(df, encode_map=None):
    if not encode_map:
        encode_map = {'low': 0.25, 'moderate': 0.5,
                      'high': 0.75, 'very high': 1}

    df['gamma_ray'] = df['gamma_ray'].map(encode_map)
    return df


def split_train_test(df, split=600, enc_type='simple'):
    if enc_type == 'simple':
        df_ = simple_encode(df.copy())
        X_train, y_train = df_.iloc[split:].drop(
            columns=['target'], inplace=False), df_.iloc[split:]['target']
        X_test, y_test = df_.iloc[:split].drop(
            columns=['target'], inplace=False), df_.iloc[:split]['target']
    elif enc_type == 'onehot':
        df_ = one_hot_encode(df.copy(), ['gamma_ray'])
        X_train, y_train = df_.iloc[split:].drop(
            columns=['target'], inplace=False), df_.iloc[split:]['target']
        X_test, y_test = df_.iloc[:split].drop(
            columns=['target'], inplace=False), df_.iloc[:split]['target']

    return X_train, y_train, X_test, y_test


def save_model(model, acc, enc=''):
    json_arch = model.to_json()
    if enc:
        model.save('./models/model_{}_{}.h5'.format(acc, enc))
        with open('./arch/model_{}_{}.json'.format(acc, enc), 'w') as out:
            out.write(json_arch)
    else:
        model.save('./models/model_{}.h5'.format(acc))
        with open('./arch/model_{}.json'.format(acc), 'w') as out:
            out.write(json_arch)


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def get_preds_on_test(model, enc='simple'):
    df = pd.read_csv('./robot_data/test_data.csv')
    df = df.drop(columns=['year', 'target'])

    if enc == 'simple':
        df = simple_encode(df.copy())

    elif enc == 'onehot':
        df = one_hot_encode(df.copy(), ['gamma_ray'])

    y_pred = model.predict(df)
    return y_pred


def create_submition(model, enc='simple', subm_name=None):
    df = pd.read_csv('./robot_data/test_data.csv')
    years = df['year']

    y_pred = get_preds_on_test(model)
    y_pred = y_pred.reshape(1000)

    d = {'year': years.values, 'target': y_pred}
    ans = pd.DataFrame(d)
    ans = ans.set_index('year')

    subm_name = subm_name if subm_name else 'submission_.csv'
    ans.to_csv(subm_name)


def np_r2_score(v_true, v_pred):
    ssres = np.sum(np.square(v_true - v_pred))
    sstot = np.sum(np.square(v_true - np.mean(v_true)))
    return 1 - ssres / sstot


def get_data():
    df = pd.read_csv('robot_data/train_data.csv')
    df.drop(columns=['year'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df
