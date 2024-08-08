from flask import Flask, render_template, abort,request,redirect,url_for,flash,jsonify
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email
import driver
import Rainfall
import alerter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

app.secret_key='5791628bb0b13ce0c676dfde280ba245' 
#app.config['SECRET_KEY']='5791628bb0b13ce0c676dfde280ba245'
 
# @app.route('/')
# def home():
#     return render_template('main.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


@app.route('/refreshFlood')
def refreshFlood():
    alerter.water_level_predictior()#To refresh the flood warning data
    return redirect(url_for('floodHome'))
@app.route('/about')
def about_team():
    return render_template('about-team.html')

@app.route('/contacts')
def contact():
    return render_template('contact.html')

@app.route('/services')
def service():
    return render_template('service.html')


@app.route('/floodHome')
def floodHome():
    res=alerter.alerting()
    for i in range(len(res)):
        res[i]='Flood ALERT for '+res[i]
    return render_template('flood_entry.html',result=res)


@app.route('/rainfallHome')
def rainfallHome():
    return render_template('rain_entry.html')


@app.route('/floodResult',methods=['POST', 'GET'])
def floodResult():
    if request.method == 'POST':
        if len(request.form['DATE'])==0:
            return redirect(url_for('floodHome'))
        else:
            user_date=request.form['DATE']
            river=request.form['SEL']
            # print("##3#######",user_date,"#####",river,"#############")
            # print(type(user_date))
            # print(type(river))
            results_dict=driver.drive(river,user_date)
            # results_dict={'Mse':0.5,
            #         'discharge':1400}
            print("-----------",type(results_dict),"----------")
            Table = []
            for key, value in results_dict.items():
                # temp = []
                # temp.extend([key,value])  #Note that this will change depending on the structure of your dictionary
                Table.append(value)
            return render_template('flood_result.html',result=Table)
    else:
        return redirect(url_for('floodHome'))
    

   # return render_template('floodResult.html')
@app.route('/rainfallResult',methods=['POST','GET'])
def rainfallResult():
    if request.method == 'POST':
        if len(request.form['Year'])==0:
            flash("Please Enter Data!!")
            return redirect(url_for('rainfallHome'))
        else:
            year=request.form['Year']
            region=request.form['SEL']
            print("##3#######",year,"#####",region,"#############")
            mae,score=Rainfall.rainfall(year,region)
            return render_template('rain_result.html',Mae=mae,Score=score)
    else:
        return redirect(url_for('rainfallHome'))
    # return render_template('rainfallResult.html')


if __name__ == '__main__':
    app.run(debug = True)