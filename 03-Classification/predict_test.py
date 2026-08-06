# Import
import requests

# Url
host = 'churn-service-env.eba-nza53frx.eu-north-1.elasticbeanstalk.com'
url = f'http://{host}/churnpredictions'

# Customer Data
customer = {
 'tenure': 12,
 'monthlycharges': 20.95,
 'totalcharges': 286.45,
 'gender': 'Female',
 'partner': 'No',
 'dependents': 'No',
 'phoneservice': 'Yes',
 'multiplelines': 'No',
 'internetservice': 'Fiber optic',
 'onlinesecurity': 'No',
 'onlinebackup': 'Yes',
 'deviceprotection': 'No',
 'techsupport': 'No',
 'streamingtv': 'Yes',
 'streamingmovies': 'Yes',
 'contract': 'Month-to-month',
 'paperlessbilling': 'Yes',
 'paymentmethod': 'Mailed check'
}

# Get and Print Prediction
response = requests.post(url, json=customer).json()
print(response)
if response['churn_prediction'] == True:
    print('The customer is predicted to leave us in the next 3 months. Send a promotional e-mail with a discount ASAP.')
else:
    print('The customer is predicted to stay with us. No immediate need to send any promotional discount.')# %%



