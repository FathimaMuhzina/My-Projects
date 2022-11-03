import pandas as pd
from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/show')
def show():
    return render_template('prediction.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if(request.method == 'POST'):
        hour=float(request.form['hour'])
        temperature=float(request.form['temperature'])
        generated=float(request.form['solar'])
        garage=float(request.form['garage'])
        living=float(request.form['living'])
        furnace=float(request.form['furnace'])
        barn=float(request.form['barn'])
        well=float(request.form['well'])
        kitchen=float(request.form['kitchen'])
        homeoffice=float(request.form['homeoffice'])
        fridge=float(request.form['fridge'])
        dishwasher=float(request.form['dishwasher'])
        microwave=float(request.form['microwave'])
        cellar=float(request.form['cellar'])
    
        data = {'hour':[hour], 'Generated':[generated], 'Dishwasher':[dishwasher], 'Home office':[homeoffice], 'Fridge':[fridge], 'Wine cellar':[cellar], 'Garage door':[garage], 'Barn':[barn], 'Well':[well], 'Microwave':[microwave], 'Living room':[living], 'temperature':[temperature], 'Furnace':[furnace], 'Kitchen':[kitchen]}
  
       
        df = pd.DataFrame(data)
        
        scaled=StandardScaler()
        df_scaled = pd.DataFrame(scaled.fit_transform(df),columns = df.columns)
      
        pred = model.predict(df_scaled)
        #output = (pred*1.058207)+0.858962
        #output=round(output.item(),4)
    
        return render_template ('result.html',prediction_text="Overall energy consumption is {} kW".format(pred))

if __name__=='__main__':
    app.run(port=8000)