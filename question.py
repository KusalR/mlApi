import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

QuestionJson = pd.read_json(r'https://helankadaapp-default-rtdb.firebaseio.com/Questions.json', orient='index')
QuestionJson.to_csv(r'Questions.csv', index=None)
ds = pd.read_csv("Questions.csv")
ds.head()

# creating the metadata column
ds['soup'] = ds['questions'] + " " + ds['keywords']
ds.head()

question_ID = ds['question_ID']
indices = pd.Series(ds.index, index=ds['question_ID'])

# column named 'soup' consisting of the metadata of the posts. column to feed TfidfVectorizer.
tf_meta = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 8), min_df=0, stop_words='english')
tfidf_matrix_meta = tf_meta.fit_transform(ds['soup'].values.astype('U'))
tfidf_matrix_meta.shape

# calculating cosine similarities
cosine_similarities_meta = cosine_similarity(tfidf_matrix_meta, tfidf_matrix_meta)
cosine_similarities_meta[0]


def item(id):
    return ds.loc[ds['question_ID'] == id]['questions'].tolist()[0]


def get_cont_recommendations(question_ID, num, indices, cosine_similarities):
    idx = indices[question_ID]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num + 1]
    # print(sim_scores)
    indices = [i[0] for i in sim_scores]
    df_return = ds.iloc[indices]
    df_return["rank"] = [i + 1 for i in range(len(df_return))]
    df_return = df_return.set_index("rank")
    df_return = df_return[["question_ID", "questions", "keywords"]]
    return df_return.to_json(orient='records')
