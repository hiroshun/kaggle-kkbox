import pandas as pd
import numpy as np
import lightgbm as lgb

train_merge = pd.read_pickle('../train_merge_df5.pkl')

mix_feature = ["source_system_tab", "source_screen_name", "source_type", "song_id", "genre_ids", "artist_name", "composer", "lyricist", 
               "language", "name", "song_country", "song_year", "song_count", "artist_count", "msno", "city", "bd", "registration_days", 
               "reg_month", "expiration_days", "exp_month", "continuation", "user_count", "userartist_count"] + ["pca_" + str(i) for i in range(5)]

n_train = int(train_merge.shape[0] * 0.8)

def valid_fit(feature, params):

    train = train_merge.loc[:n_train - 1, feature + ['target']].sample(frac = 1)
    valid = train_merge.loc[n_train:, feature + ['target']]

    X_train = train.drop(['target'], axis=1)
    y_train = train.target.values

    X_valid = valid.drop(['target'], axis=1)
    y_valid = valid.target.values

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid)
    
    return lgb.train(params, train_set=lgb_train, num_boost_round=500, valid_sets=[lgb_valid], verbose_eval=20, early_stopping_rounds=20)

mix_params = {
        'application': 'binary',
        'num_leaves': 255,
        'max_depth': -1,
        'learning_rate': 0.08,
        'metric' : 'auc',
    
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1
        }

print("----- fitting mix model -----")
mix_model = valid_fit(mix_feature, mix_params)
mix_model.save_model('valid_mix_model')
