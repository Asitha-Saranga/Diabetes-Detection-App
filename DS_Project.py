import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

st.header("Diabetes Detected application")

image = Image.open("E:\\DSProject\\diabetes-detection-app\\Data Set\\img1.jfif")
st.image(image)

data = pd.read_csv("E:\\DSProject\\diabetes-detection-app\\Data Set\\diabetes.csv")
st.subheader("Data set")
st.dataframe(data)
st.subheader("Data Description")
st.write(data.iloc[:,:8].describe())

x = data.iloc[:,:8].values
y = data.iloc[:,8].values

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2,random_state=0)

model = RandomForestClassifier(n_estimators=500)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
st.subheader("Accuracy Trained Model")
st.write(accuracy_score(y_test,y_pred))

# st.text_input(label="Enter your age")
# st.slider("Set your age ", 0,100,0)
# st.text_area(label="Desribe you")
st.subheader("Enter Your Input Data: ")

preg = st.slider("pregnancy",0,20,0)
glu = st.slider("Glucose",0,20,0)
bp = st.slider("Blood Pressure",0,130,0)
sthick = st.slider("Skin Thickness",0,100,0)
ins = st.slider("Insulin",0.0,1000.0,0.0)
bmi = st.slider("BMI",0.0,70.0,0.0)
dpf = st.slider("DPF",0.000,3.000,0.000)
age = st.slider("Age",0,100,0)

input_dict = {"pregnancies":preg, "Glucose":glu,"Blood Pressure":bp,"Skin Thickness":sthick,"Insulin":ins,"BMI":bmi,"DPF":dpf,"Age":age}
ui = pd.DataFrame(input_dict,index=["User Input Values"])

st.subheader("Entered Input Data")
st.write(ui)

st.subheader("Predictions (0-Non Diabetes, 1- Diabetes)")

st.write(model.predict(ui))

# plt.imshow(image)
# plt.axis("off")
# plt.show()