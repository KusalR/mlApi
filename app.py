from flask import Flask, request

import post_recommendation
import question
from question import indices, cosine_similarities_meta

app = Flask(__name__)


@app.route('/post_recommendation', methods=['POST'])
def recommendation_posts():
    input_Uid = request.json["UserID"]
    input_UserStatus = request.json["new_User"]
    print(input_UserStatus)

    if input_UserStatus == "False":
        res = post_recommendation.get_col_recommendations_m2(input_Uid, 5, 3)
    else:
        res = post_recommendation.get_top_posts(input_Uid, 100)

    return res.to_json(orient='records')


@app.route('/question_recommendation', methods=['POST'])
def recommendation_Questions():
    input_Qid = request.json["QuestionID"]
    res = question.get_cont_recommendations(input_Qid, 10, indices, cosine_similarities_meta)
    return res


if __name__ == '__main__':
    app.run()
