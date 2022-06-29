import joblib
from flask import Flask, render_template, request
import numpy as np


app = Flask(__name__)
model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')

dummy_data={
    '0': [1, 0],
    '1': [0, 1],
    '2': [0,  0]
}


@app.route('/', methods=['GET'])
def home():
    return render_template('penguinv2.html')

@app.route('/back', methods=['GET'])
def benguinv2():
    return render_template('benguinv2.html')


@app.route('/predict', methods=['GET'])
def predict():
    data1 = [
        float(request.args['culmen_length_mm']),
        float(request.args['culmen_depth_mm']),
        float(request.args['flipper_length_mm']),
        float(request.args['body_mass_g'])
    ]
    data4=[
        float(request.args['sex_MALE'])

    ]
    data1= np.reshape(data1,(1,-1))
    data1 = scaler.transform(data1)
    data2 = []

    for v in data1:
        for x in v:
            data2.append(float(x))

    data3 = dummy_data[request.args['caa']]
    data3 += data4
    data=data2+data3

    prediction = round(model.predict([data])[0])
    x=''
    if prediction ==0:
        x='Adelie'
    elif prediction==1:
        x='Gentoo'

    else:
        x='Chinstrap'


    return render_template('result.html', the_predicted_type_is=x)



if __name__ == "__main__":
    app.run(debug=True)