from crypt import methods
from flask import Flask, render_template, request, url_for
from helper_function import run_model

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])

def iris_prediction():
    
    if request.method == 'POST':
        
        sp_len = float(request.form['feature 1'])
        sp_wid = float(request.form['feature 2'])
        pt_len = float(request.form['feature 3'])
        pt_wid = float(request.form['feature 4'])

        list_features = [sp_len, sp_wid, pt_len, pt_wid]
        name = run_model(list_features)

        return render_template('main.html', pred = name)

    return render_template('main.html')

@app.route('/play', methods=['GET', 'POST'])

def video():
    return render_template('video.html')
    

if __name__ == '__main__':
    app.run(debug=True, port = 8732)