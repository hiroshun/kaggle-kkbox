import pandas as pd
import numpy as np
import lightgbm as lgb

train_merge = pd.read_pickle('../train_merge_df5.pkl')

mix_feature = ["source_system_tab", "source_screen_name", "source_type", "song_id", "genre_ids", "artist_name", "composer", "lyricist", 
               "language", "name", "song_country", "song_year", "song_count", "artist_count", "msno", "city", "bd", "registration_days", 
               "reg_month", "expiration_days", "exp_month", "continuation", "user_count", "userartist_count"] + ["pca_" + str(i) for i in range(5)]
song_feature = ["song_id", "genre_ids", "artist_name", "composer", "lyricist", "language", "name", "song_country", "song_year", "song_length", 
                "song_count", "artist_count"]
user_feature = ["msno", "city", "bd", "gender", "expiration_days", "registration_days", "reg_month", "exp_month",
                "registered_via", "continuation", "user_count"]

n_train = int(train_merge.shape[0] * 0.8)

def fit(feature, params, n_iter):
    
    train = train_merge[feature + ['target']].sample(frac = 1)
    X_train = train.drop(['target'], axis=1)
    y_train = train.target.values

    lgb_train = lgb.Dataset(X_train, y_train)
    
    return lgb.train(params, train_set=lgb_train, num_boost_round=n_iter, valid_sets=[lgb_train], verbose_eval=20)


song_params = {
        'application': 'binary',
        'num_leaves': 63,
        'max_depth': -1,
        'learning_rate': 0.08,
        'metric' : 'auc',
    
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1
        }
print("----- fitting song model -----")
song_model = fit(song_feature, song_params, 70)
song_model.save_model('song_model')

user_params = {
        'application': 'binary',
        'num_leaves': 4095,
        'max_depth': -1,
        'learning_rate': 0.08,
        'metric' : 'auc',
    
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1
        }
print("----- fitting user model -----")
user_model = fit(user_feature, user_params, 15)
user_model.save_model('user_model')

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
mix_model = fit(mix_feature, mix_params, 114)
mix_model.save_model('mix_model')

print("done.")
