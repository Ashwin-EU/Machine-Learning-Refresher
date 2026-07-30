# Import 
import pickle
from flask import Flask
from flask import request
from flask import jsonify

# Load Model
model_file = "model_C=1.bin"
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

#Flask app
app = Flask('Churn')
@app.route('/churnpredictions', methods = ['POST'])
def predict():
    customer = request.get_json()
    customer_transformed = dv.transform([customer])
    y_pred = model.predict_proba(customer_transformed)[0,1]
    churn = y_pred >= 0.5
    result = {
        'churn_probability': float(y_pred),
        'churn_prediction': bool(churn)
    }
    return jsonify(result)
    
if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port=9696)
