# -*- coding: utf-8 -*- 

from gensim.models import Word2Vec
from flask import Flask, request, jsonify

# 테스트를 위해 CORS를 처리한다.
from flask_cors import CORS

# 플라스크 웹 서버 객체를 생성한다.
app = Flask(__name__)
CORS(app)

def init():
    global model
    model = Word2Vec.load('neighbor_model')

# def process_from_text(neighbor):
#     model.most_similar(neighbor)

@app.route("/estimator", methods=['GET', 'POST'])
def estimator():
    content = request.json
    # process_from_text(content['neighbor'])
    # result = {'result': }
    return jsonify(model.most_similar(content['neighbor']))

if __name__ == '__main__':
    init()
    app.run('0.0.0.0', port=5000, threaded=True)    # 처리 속도 향상을 위해 쓰레드를 적용한다.

