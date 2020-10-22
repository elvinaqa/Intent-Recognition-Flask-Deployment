import pandas as pd
from flask import Flask, jsonify,render_template, request
import pickle
import warnings
warnings.filterwarnings('ignore')

# load model
sgd = pickle.load(open('pickles/SGD_Inscope.pkl','rb'))
easy_ensemble = pickle.load(open('pickles/Easy_ensemble_in_out.pkl','rb'))
brfc = pickle.load(open('pickles/BRFC_in_out.pkl','rb'))

cv_in = pickle.load(open('pickles/CountVect_Inscope.pkl','rb'))
cv_out = pickle.load(open('pickles/CountVect_Outscope.pkl','rb'))
encoder = pickle.load(open('pickles/LabelEncoder.pkl','rb'))
# app
app = Flask(__name__)

# routes

@app.route('/', methods=['GET','POST'])
def homepage():
    return render_template('index.html')

@app.route('/results', methods=['GET','POST'])
def results():
    # get data
    data = request.form['command']

    # convert data into dataframe
    data_df = pd.DataFrame({'text':[data]})
    X = cv_in.transform(data_df.text)

    #predict
    result = sgd.predict(X)
    result = encoder.inverse_transform(result)

    X2 = cv_out.transform(data_df.text)
    result2 = brfc.predict(X2)
    res = [result[0], str(result2[0]),data]
    print(res)
    # return data
    return render_template('results.html',predictions=res)

if __name__ == '__main__':
    app.run(host= '127.0.0.1',port = 5000, debug=True)
