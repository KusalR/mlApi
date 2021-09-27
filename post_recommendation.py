import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

"""# Loading Data

#### post_data.csv
"""
postJson = pd.read_json(r'https://helankadaapp-default-rtdb.firebaseio.com/Posts.json', orient='index')
postJson.to_csv(r'post_data.csv',  index=None)
posts = pd.read_csv('post_data.csv')
posts.rename(columns={'_id': 'postid', ' post_type': 'post_type'}, inplace=True)
posts.category.fillna("General", inplace=True)
posts.head()

print(posts.shape)
posts.info()

"""#### user_data.csv"""

userJson = pd.read_json(r'https://helankadaapp-default-rtdb.firebaseio.com/Users.json', orient='index')
userJson.to_csv(r'user_data.csv', index=None)
users = pd.read_csv('user_data.csv')
users.rename(columns={'id': 'user_id'}, inplace=True)
users.head()

print(users.shape)
users.info()

"""#### view_data.csv"""
viewJson = pd.read_json(r'https://helankadaapp-default-rtdb.firebaseio.com/Views.json')
viewJson.to_csv(r'view_data.csv', index=None)
views = pd.read_csv('view_data.csv')
views.head()

print(views.shape)
views.info()

"""#### merging the csv files"""

df = views.merge(posts, on='postid', how='left')
df = df.merge(users, on='user_id', how='left')

df.shape

df.info()

df.head()


"""# Recommendation process"""


def item(id):
    return posts.loc[posts['postid'] == id]['description'].tolist()[0]


def get_cont_recommendations(postid, num, indices, cosine_similarities):
    idx = indices[postid]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num + 1]
    # print(sim_scores)
    indices = [i[0] for i in sim_scores]
    print("Recommending " + str(num) + " posts similar to \"" + item(postid) + "\" ...")
    df_return = posts.iloc[indices]
    df_return["rank"] = [i + 1 for i in range(len(df_return))]
    df_return = df_return.set_index("rank")
    df_return = df_return[["postid", "description", "category"]]
    return df_return


postid = posts['postid']
indices = pd.Series(posts.index, index=posts['postid'])

"""#### Content based filrering with only category"""

# generating tfidf vectors
tf_category = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix_category = tf_category.fit_transform(posts['category'])
tfidf_matrix_category.shape

tf_category.get_feature_names()

# calculating cosine similarities
cosine_similarities_category = cosine_similarity(tfidf_matrix_category, tfidf_matrix_category)
cosine_similarities_category[0]

# Attempt 1
# get_cont_recommendations(10260109, 10, indices, cosine_similarities_category)

# Attempt 2
# get_cont_recommendations(39550285, 10, indices, cosine_similarities_category)

"""#### Content based filtering with metadata"""

# creating the metadata column
posts['soup'] = posts['description'] + " " + posts['category']
posts.head()

# column named 'soup' consisting of the metadata of the posts. column to feed TfidfVectorizer.
tf_meta = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 8), min_df=0, stop_words='english')
tfidf_matrix_meta = tf_meta.fit_transform(posts['soup'])
tfidf_matrix_meta.shape

# calculating cosine similarities
cosine_similarities_meta = cosine_similarity(tfidf_matrix_meta, tfidf_matrix_meta)
cosine_similarities_meta[0]

# Attempt 3
# get_cont_recommendations(995833095, 10, indices, cosine_similarities_meta)

# Attempt 4
# get_cont_recommendations(521413892, 10, indices, cosine_similarities_meta)

"""### comparison"""

# get_cont_recommendations(10260109, 10, indices, cosine_similarities_category)
#
# get_cont_recommendations(10260109, 10, indices, cosine_similarities_meta)
#
# get_cont_recommendations(505078124, 10, indices, cosine_similarities_category)
#
# get_cont_recommendations(505078124, 10, indices, cosine_similarities_meta)

"""### Collaborative Filtering"""

df_orginal = df.copy()

# creating user engagement data frame
df_user_unq_post = df.groupby(["user_id"]).agg({"postid": 'nunique'})
df_user_unq_post.columns = ["num_diff_posts"]
df_user_unq_post.reset_index(inplace=True)
df_user_unq_post

df_user_unq_post["num_diff_posts"].describe()

thr_user = 0  # if you need to filterout less engaging users you can give a threshold here
selected_user_ids = list(
    df_user_unq_post[df_user_unq_post["num_diff_posts"] >= thr_user]["user_id"])  # selecting user ids

# creating post engagement data frame
df_post_unq_user = df.groupby(["postid"]).agg({"user_id": 'nunique'})
df_post_unq_user.columns = ["num_diff_users"]
df_post_unq_user.reset_index(inplace=True)
df_post_unq_user

df_post_unq_user["num_diff_users"].describe()

thr_post = 0  # if you need to filterout less engaging posts you can give a threshold here
selected_postids = list(df_post_unq_user[df_post_unq_user["num_diff_users"] >= thr_post]["postid"])

# filtering less engaging users and posts
df["sel_users"] = df["user_id"].apply(lambda x: int(x in selected_user_ids))
df["sel_posts"] = df["postid"].apply(lambda x: int(x in selected_postids))
df = df[df["sel_users"] == 1]
df = df[df["sel_posts"] == 1]
df = df[["user_id", "postid"]]

# creating the rating data frame
df_rating = df[["user_id", "postid"]]
df_rating["Quantity"] = 1
df_rating = df_rating.groupby(["user_id", "postid"]).agg({'Quantity': 'sum'})
df_rating.reset_index(inplace=True)
df_rating = df_rating.merge(df_post_unq_user, on='postid', how='left')

df_rating.head()

# creating the post correlation data frame
df_post_pivot = df_rating.pivot(index="user_id", columns="postid", values='Quantity').fillna(0)
df_post_corr = df_post_pivot.corr(method='spearman', min_periods=5)
df_post_corr

# creating the user correlation data frame
df_user_pivot = df_rating.pivot(index="postid", columns="user_id", values='Quantity').fillna(0)
df_user_corr = df_user_pivot.corr(method='spearman', min_periods=5)
df_user_corr


def get_col_recommendations_m1(postid, top_n):
    df_post = df_post_corr.loc[postid]
    df_post_sort = df_post.sort_values(ascending=False).head(top_n + 1)
    df_post_sort = pd.merge(df_post_sort, df_orginal.drop_duplicates(subset=["postid"]), how='left', on="postid")
    sel_item = list(df_post_sort[df_post_sort["postid"] == postid]["description"])[0]
    print("Recommending " + str(top_n) + " posts similar to \"" + sel_item + "\" ...")
    df_ = df_post_sort[df_post_sort["postid"] != postid]
    df_["rank"] = [i + 1 for i in range(len(df_))]
    df_ = df_.set_index("rank")
    df_ = df_[["postid", "description", "category"]]
    return (df_)


def sim_users(user_id, top_n):
    df_user = df_user_corr.loc[user_id]
    df_user_sort = df_user.sort_values(ascending=False).head(top_n + 1)
    user_ids = list(df_user_sort.index)
    user_ids.remove(user_id)
    return (user_ids)


def get_top_posts(user_id, top_n):
    df_select = df_rating[df_rating["user_id"] == user_id].sort_values(['Quantity', 'num_diff_users'],
                                                                       ascending=False).head(top_n)
    df_select = posts.set_index("postid").loc[list(df_select["postid"])].reset_index()[
        ["postid", "description", "category"]]
    return df_select


def get_col_recommendations_m2(user_id, n_users, n_posts):
    users_list = sim_users(user_id, n_users)
    df_ = pd.DataFrame()
    for id_ in users_list:
        df_ = pd.concat([df_, get_top_posts(id_, n_posts)], ignore_index=True)
    df_ = df_.drop_duplicates(subset=['postid'])
    df_["rank"] = [i + 1 for i in range(len(df_))]
    df_ = df_.set_index("rank")
    df_ = df_[["postid", "description", "category"]]
    return df_


# get_col_recommendations_m1(10164988, 10)


# get_col_recommendations_m2("5eece14ffc13ae66090001f3", 5, 3)

print('***********')
