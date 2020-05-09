import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

ridgereg=pickle.load(open('ridgereg.pkl','rb'))








#stacked_averaged_models=pickle.load(open('stacked_averaged_models.pkl','rb'))


@app.route('/')
def home():
    return render_template('gorab2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()] 
    final_features = [np.array(int_features)]
    prediction = ridgereg.predict(final_features)
    #prediction2 = model_lgb.predict(final_features)
    #prediction = stacked_averaged_models.predict(final_features)
    #prediction=mean()
    output = round(prediction[0], 3)

    return render_template('gorab2.html', prediction_text='the predicted price for your choice of features is  {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = ridgereg.predict([np.array(list(data.values()))])


    output = prediction    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
