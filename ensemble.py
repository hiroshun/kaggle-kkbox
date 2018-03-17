import pandas as pd
import numpy as np
import lightgbm as lgb

print("loading dataset...")
train = pd.read_pickle('../train_merge_df5.pkl')
test = pd.read_pickle('../test_merge_df5.pkl')

print("loading model...")
song_model = lgb.Booster(model_file="song_model")
user_model = lgb.Booster(model_file="user_model")
mix_model = lgb.Booster(model_file="mix_model")

mix_feature = ["source_system_tab", "source_screen_name", "source_type", "song_id", "genre_ids", "artist_name", "composer", "lyricist", 
               "language", "name", "song_country", "song_year", "song_count", "artist_count", "msno", "city", "bd", "registration_days", 
               "reg_month", "expiration_days", "exp_month", "continuation", "user_count", "userartist_count"] + ["pca_" + str(i) for i in range(5)]
song_feature = ["song_id", "genre_ids", "artist_name", "composer", "lyricist", "language", "name", "song_country", "song_year", "song_length", 
                "song_count", "artist_count"]
user_feature = ["msno", "city", "bd", "gender", "expiration_days", "registration_days", "reg_month", "exp_month",
                "registered_via", "continuation", "user_count"]

print("predicting...")

X_song = test[song_feature]
X_user = test[user_feature]
X_mix = test[mix_feature]

test["song_pred"] = song_model.predict(X_song)
test["user_pred"] = user_model.predict(X_user)
test["mix_pred"] = mix_model.predict(X_mix)

print("dividing into segments...")
# 曲を学習済みか(trainで出現したか)
known_song = list(set(test.song_id.unique()) & set(train.song_id.unique()))
known_song_df = pd.DataFrame(index=test.song_id.unique())
known_song_df["know_song"] = False
known_song_df.loc[known_song, "know_song"] = True
test = pd.merge(test, known_song_df, how='left', left_on="song_id", right_index=True)

# 曲を学習済みか(trainで出現したか)
known_user = list(set(test.msno.unique()) & set(train.msno.unique()))
known_user_df = pd.DataFrame(index=test.msno.unique())
known_user_df["know_user"] = False
known_user_df.loc[known_user, "know_user"] = True
test = pd.merge(test, known_user_df, how='left', left_on="msno", right_index=True)

test.msno = test.msno.astype("category")
test.song_id = test.song_id.astype("category")

s1 = test[test.know_user & test.know_song]
s2 = test[~test.know_user & test.know_song]
s3 = test[test.know_user & ~test.know_song]
s4 = test[~test.know_user & ~test.know_song]

print("ensembling...")

def ensemble_predict(data, w_song, w_user, w_mix):
    return (data.song_pred * w_song + data.user_pred * w_user + data.mix_pred * w_mix) / 100

s1_pred = ensemble_predict(s1, 0.03, 0.13, 0.84)
s2_pred = ensemble_predict(s2, 0.06, 0.19, 0.75)
s3_pred = ensemble_predict(s3, 0.50, 0.00, 0.50)
s4_pred = ensemble_predict(s4, 0.27, 0.12, 0.61)

test["en_pred"] = pd.concat([s1_pred, s2_pred, s3_pred, s4_pred])

print("saving...")
submit = pd.DataFrame(columns=['id', 'target'])
submit.id = test['id'].values
submit.target = test["en_pred"].values
submit.to_csv('en5.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

print("done.")
