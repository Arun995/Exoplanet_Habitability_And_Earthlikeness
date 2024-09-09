from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
habitability_model = joblib.load('model/habitability_model.pkl')
earth_likeness_model = joblib.load('model/earth_likeness_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    pl_orbper = float(request.form['pl_orbper'])
    pl_orbsmax = float(request.form['pl_orbsmax'])
    pl_eqt = float(request.form['pl_eqt'])
    pl_orbeccen = float(request.form['pl_orbeccen'])
    pl_rade = float(request.form['pl_rade'])
    pl_masse = float(request.form['pl_masse'])
    pl_dens = float(request.form['pl_dens'])
    st_teff = float(request.form['st_teff'])
    pl_orbincl = float(request.form['pl_orbincl'])
    pl_ratror = float(request.form['pl_ratror'])

    # Prepare input for habitability model (8 features)
    habitability_input = [ pl_orbsmax, pl_eqt, pl_orbeccen, pl_rade, pl_masse, pl_dens, st_teff, pl_ratror]

    # Prepare input for earth likeness model (9 features)
    earth_likeness_input = [pl_orbper, pl_orbsmax, pl_eqt, pl_orbeccen, pl_rade, pl_masse, pl_dens, pl_orbincl, pl_ratror]

    # Predict habitability and earth likeness percentages
    habitability_percentage = habitability_model.predict([habitability_input])[0]
    earth_likeness_percentage = earth_likeness_model.predict([earth_likeness_input])[0]
    
    # Render result template
    return render_template('result.html', 
                           habitability=habitability_percentage, 
                           earth_likeness=earth_likeness_percentage)

if __name__ == '__main__':
    app.run(debug=True)
