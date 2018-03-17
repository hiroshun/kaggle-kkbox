import pandas as pd
import numpy as np

prefix = ""

print("reading csv...")
# csv読み込み
train = pd.read_csv(prefix + "data/train.csv", dtype={'target' : np.uint8, 
                                        'msno' : 'category',
                                        'source_system_tab' : 'category',
                                        'source_screen_name' : 'category',
                                        'source_type' : 'category',
                                        'song_id' : 'category'})
test = pd.read_csv(prefix + "data/test.csv", dtype={'msno' : 'category',
                                      'source_system_tab' : 'category',
                                      'source_screen_name' : 'category',
                                      'source_type' : 'category',
                                      'song_id' : 'category'})
songs = pd.read_csv(prefix + "data/songs.csv", dtype={'genre_ids': 'category',
                                        'language' : 'category',
                                        'song_id' : 'category'})
song_extra = pd.read_csv(prefix + "data/song_extra_info.csv")
members = pd.read_csv(prefix + "data/members.csv", dtype={'city' : 'category',
                                            'gender' : 'category',
                                            'registered_via' : 'category'})

print("processing songs...")
# songsに対する処理
songs_merge = pd.merge(songs, song_extra, how='left', on='song_id')

# isrcから国のデータを取り出す
def get_country_by_isrc(isrc):
    return str(isrc)[:2]
songs_merge["song_country"] = songs_merge.isrc.dropna().apply(get_country_by_isrc)

# isrcから曲のリリース年を取得
def get_year_by_isrc(isrc):
    yy = int(str(isrc)[5:7])
    if yy > 17 and yy <= 39: # 18〜39は不正値
        return np.nan
    elif yy > 39:
        return 1900 + yy
    else:
        return 2000 + yy
songs_merge["song_year"] = songs_merge.isrc.dropna().apply(get_year_by_isrc)

songs_merge.drop(["isrc"], axis=1, inplace=True)

# membersに対する処理
print("processing members...")

# 日付を連続的に変換
import datetime
def to_days(str_date):
    return (datetime.datetime.strptime(str(str_date), '%Y%m%d') - datetime.datetime(1970,1,1)).days
members["registration_days"] = members.registration_init_time.apply(to_days)
members["expiration_days"] = members.expiration_date.apply(to_days)

# 登録、失効月をそれぞれ抽出
members["reg_month"] = members.registration_init_time.apply(lambda s: datetime.datetime.strptime(str(s), '%Y%m%d').month)
members["exp_month"] = members.expiration_date.apply(lambda s: datetime.datetime.strptime(str(s), '%Y%m%d').month)

# 継続日数
members['continuation'] = members.expiration_days - members.registration_days

# 年齢が0以下と80より大きいものは削除
members.loc[(members.bd <= 0) | (members.bd > 80), "bd"] = np.nan

members.drop(["registration_init_time", "expiration_date"], axis=1, inplace=True)

print("merging dataframe...")
# 各DataFrameのマージ
train_merge = pd.merge(train, songs_merge, how='left', on='song_id')
train_merge = pd.merge(train_merge, members, how='left', on='msno')
test_merge = pd.merge(test, songs_merge, how='left', on='song_id')
test_merge = pd.merge(test_merge, members, how='left', on='msno')

members = members[members.continuation >= 0]
train_merge = train_merge[train_merge.continuation >= 0]

print("counting song, user, artist...")
# 曲ごとのユーザ再生回数のカウント
song_count = train.song_id.value_counts().add(test.song_id.value_counts(), fill_value=0)
train_merge['song_count'] = pd.merge(pd.DataFrame(train.song_id), pd.DataFrame(song_count), 
                                     how='left', left_on='song_id', right_index=True, suffixes=('', '_count'))['song_id_count']
test_merge['song_count'] = pd.merge(pd.DataFrame(test.song_id), pd.DataFrame(song_count), 
                                     how='left', left_on='song_id', right_index=True, suffixes=('', '_count'))['song_id_count']
train_merge['song_count'] = train_merge['song_count'].astype(np.float64)
test_merge['song_count'] = test_merge['song_count'].astype(np.float64)

# アーティストごとのユーザ再生回数
artist_count = train_merge.artist_name.value_counts().add(test_merge.artist_name.value_counts(), fill_value=0)
train_merge['artist_count'] = pd.merge(pd.DataFrame(train_merge.artist_name), pd.DataFrame(artist_count), how='left', 
                                       left_on='artist_name', right_index=True, suffixes=('', '_count'))['artist_name_count']
test_merge['artist_count'] = pd.merge(pd.DataFrame(test_merge.artist_name), pd.DataFrame(artist_count), how='left', 
                                      left_on='artist_name', right_index=True, suffixes=('', '_count'))['artist_name_count']
train_merge['artist_count'] = train_merge['artist_count'].astype(np.float64)
test_merge['artist_count'] = test_merge['artist_count'].astype(np.float64)

# ユーザごとの曲再生回数
user_count = train_merge.msno.value_counts().add(test_merge.msno.value_counts(), fill_value=0)
train_merge['user_count'] = pd.merge(pd.DataFrame(train_merge.msno), pd.DataFrame(user_count), how='left', 
                                       left_on='msno', right_index=True, suffixes=('', '_count'))['msno_count']
test_merge['user_count'] = pd.merge(pd.DataFrame(test_merge.msno), pd.DataFrame(user_count), how='left', 
                                      left_on='msno', right_index=True, suffixes=('', '_count'))['msno_count']
train_merge['user_count'] = train_merge['user_count'].astype(np.float64)
test_merge['user_count'] = test_merge['user_count'].astype(np.float64)

train_merge["userartist"] = train_merge.msno.astype(str) + train_merge.artist_name.astype(str)
test_merge["userartist"] = test_merge.msno.astype(str) + test_merge.artist_name.astype(str)
userartist_count = train_merge.userartist.value_counts().add(test_merge.userartist.value_counts(), fill_value=0)
train_merge['userartist_count'] = pd.merge(pd.DataFrame(train_merge.userartist), pd.DataFrame(userartist_count), how='left', 
                                       left_on='userartist', right_index=True, suffixes=('', '_count'))['userartist_count']
test_merge['userartist_count'] = pd.merge(pd.DataFrame(test_merge.userartist), pd.DataFrame(userartist_count), how='left', 
                                      left_on='userartist', right_index=True, suffixes=('', '_count'))['userartist_count']
train_merge['userartist_count'] = train_merge['userartist_count'].astype(np.float64)
test_merge['userartist_count'] = test_merge['userartist_count'].astype(np.float64)
    
import gc
del members, songs, songs_merge, train, test, song_count, artist_count, user_count; gc.collect();

print("converting feature type...")
# 型の変換
for col in train_merge.columns:
    if train_merge[col].dtype == object:
        train_merge[col] = train_merge[col].astype('category')
        test_merge[col] = test_merge[col].astype('category')
for col in ['language', 'city', 'registered_via']:
    train_merge[col] = train_merge[col].astype('category')
    test_merge[col] = test_merge[col].astype('category')

for col in train_merge.columns:
    if str(train_merge[col].dtype) == "category":
        train_merge[col] = train_merge[col].cat.add_categories("NaN").fillna("NaN")
        test_merge[col] = test_merge[col].cat.add_categories("NaN").fillna("NaN")
        
# user-artist行列のPCA
print("user-artist PCA")
from sklearn import preprocessing
from sklearn.decomposition import PCA

data = pd.concat([train_merge[["msno", "artist_name"]], test_merge[["msno", "artist_name"]]], ignore_index=True)
label_msno = preprocessing.LabelEncoder()
label_artist = preprocessing.LabelEncoder()
label_msno.fit(data.msno.unique())
data["user_label"] = label_msno.transform(data.msno.values)
label_artist.fit(data.artist_name.unique())
data["artist_label"] = label_artist.transform(data.artist_name.values)
# 行: ユーザ名, 列: アーティスト
matrix = np.zeros((data.user_label.max()+1, data.artist_label.max()+1))
matrix[data.user_label, data.artist_label] = 1
# PCA
pca = PCA(n_components=5)
pca.fit(matrix.T)
user_pca = pca.components_
user_df = pd.DataFrame(np.arange(data.user_label.max()+1), columns=['label'])
user_df['msno'] = label_msno.inverse_transform(user_df.label)
for i, a in enumerate(user_pca):
    user_df["pca_" + str(i)] = a[user_df.label]
    
train_merge = pd.merge(train_merge, user_df, how='left', on='msno')
test_merge = pd.merge(test_merge, user_df, how='left', on='msno')
train_merge["msno"] = train_merge["msno"].astype('category')
test_merge["msno"] = test_merge["msno"].astype('category')

# 保存
save_name_train_df = 'train_merge_df5.pkl' 
save_name_test_df = 'test_merge_df5.pkl' 

train_merge.to_pickle(save_name_train_df)
print("save train_merge dataframe {}".format(save_name_train_df))
test_merge.to_pickle(save_name_test_df)
print("save test_merge dataframe {}".format(save_name_test_df))

print("done.")

