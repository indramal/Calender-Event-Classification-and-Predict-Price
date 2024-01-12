import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pandas as pd
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import pipeline
import torch
import io
import base64
import pickle

from fastapi import FastAPI
from mangum import Mangum
import uvicorn

from aws_lambda_powertools import Logger 
logger = Logger()

if not firebase_admin._apps:
  cred = credentials.Certificate("bill-extractor-586a1-firebase-adminsdk-bnhw2-c376f19014.json")
  firebase_admin.initialize_app(cred)
db = firestore.client()

useridv = ""
colname = ""
calendarcolname = ""

model_checkpoint = "./Model" # NLP Model Location
regmodels_checkpoint = "./Regression_Models/" # Regression Model Location

def strtoint(data,cat):
  logger.info("strtoint function running...")
  if(cat == "gender"):
    if(data == "m"):
      return 1
    else:
      return 0
  elif(cat == "age"):
    if(data < 18):
      return 3
    elif(data <= 25):
      return 0
    elif(data <= 30):
      return 1
    elif(data < 40):
      return 2
  elif(cat == "province"):
    if(data == "Central"):
      return 0
    elif(data == "Eastern"):
      return 1
    elif(data == "North Central"):
      return 2
    elif(data == "North Western"):
      return 3
    elif(data == "Nothern"):
      return 4
    elif(data == "Sabaragamuwa"):
      return 5
    elif(data == "Southern"):
      return 6
    elif(data == "Uva"):
      return 7
    elif(data == "Western"):
      return 8
  elif(cat == "marital_status"):
    if(data == "Single"):
      return 1
    else:
      return 0
  elif(cat == "employability_status"):
    if(data == "Dependant"):
      return 0
    else:
      return 1
    
def getuserdata(colnamev, idval):
  logger.info("getuserdata function running...")

  users_ref = db.collection(colnamev)
  docs = users_ref.stream()
  for doc in docs:
    if(doc.id == idval):
      data = {
        "gender": strtoint(doc.to_dict()['gender'],"gender"),
        "age": strtoint(doc.to_dict()['age'],"age"),
        "province ": strtoint(doc.to_dict()['province'],"province"),
        "marital_status":strtoint(doc.to_dict()['marital_status'],"marital_status"),
        "employability_status":strtoint(doc.to_dict()['employability_status'],"employability_status"),
        "monthly_income":doc.to_dict()['monthly_income']
        }
      df = pd.DataFrame([data])
  return df

def textclean(text):
  logger.info("textclean function running...")
  if(pd.isna(text)):
    outputtext = float("nan")
  else:
    outputtext = BeautifulSoup(text) # Remove HTML Codes
    outputtext = outputtext.get_text() # Remove HTML Codes
    outputtext = re.sub(r'http\S+', ' ', outputtext) # Remove Links
    outputtext = re.sub(r'\S*@\S*\s?', ' ', outputtext) # Remove Emails
    outputtext = re.sub(r"[^A-Za-z0-9' ]+", ' ', outputtext) # Remove unwnated things
  return outputtext 

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

def predictcat(textdata):
  logger.info("predictcat function running...")
  classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
  outputcat = classifier(textdata)
  return outputcat

def predicprice(modelname,userdata):
  logger.info("predicprice function running...")
  loaded_model = pickle.load(open(regmodels_checkpoint+modelname, 'rb'))
  y_pred = loaded_model.predict(userdata)
  return y_pred

def addtocaldata(colvalcal,useridval,adddata):
  logger.info("addtocaldata function running...")
  users_ref = db.collection(colvalcal)
  users_ref.document(useridval).set(adddata, merge=True)

def caldataupdate(colnamev, useridv,colusnamev):
  logger.info("caldataupdate function running...")
  users_ref = db.collection(colnamev)
  docs = users_ref.stream()

  model_match = {"A party":"",
"Birthday party":"DecisionTreeRegressor_Birthday.sav",
"Boarding fees":"DecisionTreeRegressor_Boarding_Rental_Per_Month.sav",
"Charity":"DecisionTreeRegressor_Charity.sav",
"Coffee outing":"DecisionTreeRegressor_Coffee_Outing.sav",
"Dentist appointment":"DecisionTreeRegressor_dentist_appointment.sav",
"Dinner or lunch out":"DecisionTreeRegressor_Dinner_outing.sav",
"Doctor appointment":"DecisionTreeRegressor_doctor_appointment.sav",
"Exams":"DecisionTreeRegressor_Exam.sav",
"Get together":"DecisionTreeRegressor_Get_together.sav",
"Grocery shopping":"DecisionTreeRegressor_Grocery_Shopping.sav",
"Haircut":"DecisionTreeRegressor_Haircut.sav",
"Health checkup":"DecisionTreeRegressor_Health.sav",
"Movie night":"DecisionTreeRegressor_movie_night.sav",
"Musical show":"DecisionTreeRegressor_Musical_show.sav",
"Other":"",
"Saloon appointment":"DecisionTreeRegressor_Saloon_appointment_dressing_makeup.sav",
"Shopping outing":"DecisionTreeRegressor_Shopping.sav",
"Trips":"DecisionTreeRegressor_Trip.sav",
"Virtual event":"DecisionTreeRegressor_Virtual_event.sav",
"Visit grand parents":"DecisionTreeRegressor_Visit_grand_parents.sav",
"Wedding":"DecisionTreeRegressor_Wedding.sav",
"Workouts or Gym":"DecisionTreeRegressor_Workout_Gym.sav",}

  for doc in docs:
    if(doc.id == useridv):
      data = doc.to_dict()
      for calloc in data.keys(): 
        datacombined = data[str(calloc)]['title'] + " " + data[str(calloc)]['description']
        cleandatacombined = textclean(datacombined)
        calcat = predictcat(cleandatacombined)[0]['label']
        regmodelname = model_match[calcat]
        if(regmodelname != ""):
          userdata = getuserdata(colusnamev, useridv)
          predictval = round(predicprice(regmodelname,userdata)[0],2)
        else:   
          predictval = 0
        adddataval = {str(calloc):{"predicted_budget":predictval,"event_cat":calcat}}
        addtocaldata(colnamev,useridv,adddataval) # Update Database
  return None

app = FastAPI()
handler = Mangum(app)

@app.get("/")
async def index():
    return "Home"

@app.get("/api")
async def student_data(user_id:str,colusername:str,colcalname:str,token:str):

    key = "RRshJy4beYdlNbu"

    if(token == key):
      logger.info("INFO: Program is Running...")
      useridv = user_id #"tl5YWL9orwgWfMftlUgzDSDpm9Q2"
      colname = colusername #"user_data"
      calendarcolname = colcalname #"calendar_data"

      caldataupdate(calendarcolname, useridv,colname)
      return {"Status":"Done"}
    else:
      return {"Status":"Error"}

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
