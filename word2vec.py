# -*- coding: utf-8 -*- 
from gensim.models import Word2Vec
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS             # 테스트를 위해 CORS를 처리한다. 
from collections import OrderedDict     # 순서를 기억하는 사전형 라이브러기

import os                               # 파일에 접근하기 위한 라이브러리
import json

app = Flask(__name__, static_folder='outputs')                   # 플라스크 웹 서버 객체를 생성한다.
CORS(app)

file_data = OrderedDict()               # OrderedDict 타입의 file_data 생성

def init():
    global model
    model = Word2Vec.load('neighbor_model')     # Word2Vec 모델 loading

def process_feature(neighbor):
    js = model.most_similar(neighbor)           # 유사 관계 단어 추출
    file_data["children"] = [{"word": js}]

    path = 'outputs/{0}.json'.format(neighbor)  # json 파일 생성 위치

    with open(path, 'w') as f:
        json.dump(file_data, f, indent=1)

    
@app.route("/estimator", methods=['GET', 'POST'])
def estimator():
    content = request.json
    process_feature(content['neighbor'])
    result = {'result': True}
    return jsonify(result)

@app.route("/outputs", methods=['GET', 'POST'])
def outputs():
    neighbor_id = request.args.get('neighbor')
    return app.send_static_file(neighbor_id + '.json')

@app.route("/validate", methods=['GET', 'POST'])
def validate():
    neighbor_id = request.args.get('neighbor')
    path = "outputs/{0}.json".format(neighbor_id)
    result = {}

    # 해당 json 파일 존재하는지 확인
    if os.path.isfile(path):
        result['result'] = True
    else:
        result['result'] = False
    return jsonify(result)

if __name__ == '__main__':
    init()
    app.run('0.0.0.0', port=5000, threaded=True)    # 처리 속도 향상을 위해 쓰레드를 적용한다.

                                                             