import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template

app = Flask(__name__)

# 모델 로드
labels = ['남자', '여자', '수상한 사람']
model = load_model('./my_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    image = Image.open(request.files['file'].stream)
    image = image.resize((300, 300))
    image_numpy = np.array(image)
    X_test = image_numpy.reshape(1, 300, 300, 3)
    X_test = X_test / 255

    pred = model.predict(X_test)
    if labels[np.argmax(pred[0])] == '수상한 사람':
        result = '사진속 인물은 ' + labels[np.argmax(pred[0])] + '입니다. 마스크를 벗어주세요.' + ' (정확도 : ' + str(np.round(max(pred[0])*100, 2)) + '%)'
    else:
        result = '사진속 인물의 성별은 ' + labels[np.argmax(pred[0])] + '입니다.' + ' (정확도 : ' + str(np.round(max(pred[0])*100, 2)) + '%)'

    img = 'assets/img/' + labels[np.argmax(pred[0])] + '.jpg'
    return render_template('index2.html', path=img, data=result)

if __name__ == '__main__':
    app.debug = True
    app.run()

