import streamlit as st
import pandas as pd
import numpy as np
from streamlit.components.v1 import html
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import base64
from functions import *

st.set_page_config(layout="wide", page_title="Carbon Footprint Calculator", page_icon="./media/favicon.ico")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background = get_base64("./media/background_min.jpg")
icon2 = get_base64("./media/icon2.png")
icon3 = get_base64("./media/icon3.png")

with open("./style/style.css", "r") as style:
    css=f"""<style>{style.read().format(background=background, icon2=icon2, icon3=icon3)}</style>"""
    st.markdown(css, unsafe_allow_html=True)

def script():
    with open("./style/scripts.js", "r", encoding="utf-8") as scripts:
        open_script = f"""<script>{scripts.read()}</script> """
        html(open_script, width=0, height=0)

left, middle, right = st.columns([2, 3.5, 2])
main, comps, result = middle.tabs([" ", " ", " "])

with open("./style/main.md", "r", encoding="utf-8") as main_page:
    main.markdown(f"""{main_page.read()}""")

_, but, _ = main.columns([1, 2, 1])
if but.button("Calculate Your Carbon Footprint!", type="primary"):
    click_element('tab-1')

tab1, tab2, tab3, tab4, tab5 = comps.tabs(["👴 Personal", "🚗 Travel", "🗑️ Waste", "⚡ Energy", "💸 Consumption"])
tab_result, _ = result.tabs([" ", " "])

def component():
    tab1col1, tab1col2 = tab1.columns(2)
    height = tab1col1.number_input("Height", 0, 251, value=None, placeholder="160", help="in cm")
    weight = tab1col2.number_input("Weight", 0, 250, value=None, placeholder="75", help="in kg")
    if (weight is None) or (weight == 0): weight = 1
    if (height is None) or (height == 0): height = 1
    calculation = weight / (height / 100) ** 2
    body_type = "underweight" if (calculation < 18.5) else \
                 "normal" if ((calculation >= 18.5) and (calculation < 25)) else \
                 "overweight" if ((calculation >= 25) and (calculation < 30)) else "obese"
    sex = tab1.selectbox('Gender', ["female", "male"])
    diet = tab1.selectbox('Diet', ['omnivore', 'pescatarian', 'vegetarian', 'vegan'], help="""Omnivore: Eats both plants and animals.\nPescatarian: Consumes plants and seafood, but no other meat\nVegetarian: Diet excludes meat but includes plant-based foods.\nVegan: Avoids all animal products, including meat, dairy, and eggs.""")
    social = tab1.selectbox('Social Activity', ['never', 'often', 'sometimes'], help="How often do you go out?")

    transport = tab2.selectbox('Transportation', ['public', 'private', 'walk/bicycle'], help="Which transportation method do you prefer the most?")
    if transport == "private":
        vehicle_type = tab2.selectbox('Vehicle Type', ['petrol', 'diesel', 'hybrid', 'lpg', 'electric'], help="What type of fuel do you use in your car?")
    else:
        vehicle_type = "None"

    if transport == "walk/bicycle":
        vehicle_km = 0
    else:
        vehicle_km = tab2.slider('What is the monthly distance traveled by the vehicle in kilometers?', 0, 5000, 0, disabled=False)

    air_travel = tab2.selectbox('How often did you fly last month?', ['never', 'rarely', 'frequently', 'very frequently'], help= """
                                                                                                                             Never: I didn't travel by plane.\n
                                                                                                                             Rarely: Around 1-4 Hours.\n
                                                                                                                             Frequently: Around 5 - 10 Hours.\n
                                                                                                                             Very Frequently: Around 10+ Hours. """)

    waste_bag = tab3.selectbox('What is the size of your waste bag?', ['small', 'medium', 'large', 'extra large'])
    waste_count = tab3.slider('How many waste bags do you trash out in a week?', 0, 10, 0)
    recycle = tab3.multiselect('Do you recycle any materials below?', ['Plastic', 'Paper', 'Metal', 'Glass'])

    heating_energy = tab4.selectbox('What power source do you use for heating?', ['natural gas', 'electricity', 'wood', 'coal'])

    for_cooking = tab4.multiselect('What cooking systems do you use?', ['microwave', 'oven', 'grill', 'airfryer', 'stove'])
    energy_efficiency = tab4.selectbox('Do you consider the energy efficiency of electronic devices?', ['No', 'Yes', 'Sometimes'])
    daily_tv_pc = tab4.slider('How many hours a day do you spend in front of your PC/TV?', 0, 24, 0)
    internet_daily = tab4.slider('What is your daily internet usage in hours?', 0, 24, 0)

    shower = tab5.selectbox('How often do you take a shower?', ['daily', 'twice a day', 'more frequently', 'less frequently'])
    grocery_bill = tab5.slider('Monthly grocery spending in $', 0, 500, 0)
    clothes_monthly = tab5.slider('How many clothes do you buy monthly?', 0, 30, 0)

    data = {'Body Type': body_type,
            "Sex": sex,
            'Diet': diet,
            "How Often Shower": shower,
            "Heating Energy Source": heating_energy,
            "Transport": transport,
            "Social Activity": social,
            'Monthly Grocery Bill': grocery_bill,
            "Frequency of Traveling by Air": air_travel,
            "Vehicle Monthly Distance Km": vehicle_km,
            "Waste Bag Size": waste_bag,
            "Waste Bag Weekly Count": waste_count,
            "How Long TV PC Daily Hour": daily_tv_pc,
            "Vehicle Type": vehicle_type,
            "How Many New Clothes Monthly": clothes_monthly,
            "How Long Internet Daily Hour": internet_daily,
            "Energy efficiency": energy_efficiency
            }
    data.update({f"Cooking_with_{x}": y for x, y in
                 dict(zip(for_cooking, np.ones(len(for_cooking)))).items()})
    data.update({f"Do You Recyle_{x}": y for x, y in
                 dict(zip(recycle, np.ones(len(recycle)))).items()})

    return pd.DataFrame(data, index=[0])

df = component()
data = input_preprocessing(df)

sample_df = pd.DataFrame(data=sample, index=[0])
sample_df[sample_df.columns] = 0
sample_df[data.columns] = data

ss = pickle.load(open("./models/scale.sav", "rb"))
model = pickle.load(open("./models/model.sav", "rb"))
prediction = round(np.exp(model.predict(ss.transform(sample_df))[0]))

# Visualization for Carbon Footprint Result
column1, column2 = tab1.columns(2)
_, resultbutton, _ = tab5.columns([1, 1, 1])
if resultbutton.button(" ", type="secondary"):
    tab_result.image(chart(model, ss, sample_df, prediction), use_column_width="auto")
    click_element('tab-2')

# New Features: Forecasted Next Month Emission, Diet Impact Category, Emission User Type, Smart Sustainability Tips
forecasted_emission = round(prediction * 1.05)

# Diet Impact Category
diet = df['Diet'].values[0].lower()
diet_impact_map = {
    'omnivore': 'High',
    'pescatarian': 'Moderate',
    'vegetarian': 'Low',
    'vegan': 'Very Low'
}
diet_impact = diet_impact_map.get(diet, 'Unknown')

# Emission User Type
if prediction < 100:
    user_type = "🌱 Eco-Friendly"
elif prediction < 300:
    user_type = "♻️ Average Emitter"
else:
    user_type = "🔥 High Emitter"

# Smart Sustainability Tips
transport = df['Transport'].values[0].lower()
vehicle_type = df['Vehicle Type'].values[0].lower()
recycle = df.get('Do You Recyle_Paper', False)  # Check if any materials are recycled
energy_efficiency = df['Energy efficiency'].values[0].lower()
daily_tv_pc = df['How Long TV PC Daily Hour'].values[0]
internet_daily = df['How Long Internet Daily Hour'].values[0]

recommendations = []
if diet != 'vegan':
    recommendations.append("🥗 Consider reducing meat and dairy consumption.")
if transport == 'private' and vehicle_type != 'electric':
    recommendations.append("🚌 Try carpooling or switching to public transport.")
if not recycle:
    recommendations.append("♻️ Start recycling paper, plastic, glass, and metal.")
if energy_efficiency == 'no':
    recommendations.append("🔌 Choose energy-efficient appliances.")
if daily_tv_pc > 5 or internet_daily > 5:
    recommendations.append("📉 Reduce screen time to save energy.")

# Display additional information after carbon footprint visualization
tab_result.markdown(f"""
🌍 <b>Forecasted Next Month Emission:</b> {forecasted_emission} kg CO₂<br>
🥗 <b>Diet Impact Category:</b> {diet_impact}<br>
🧑‍💼 <b>Your Emission Type:</b> {user_type}<br><br>
💡 <b>Smart Sustainability Tips:</b>
<ul>
{''.join([f"<li>{tip}</li>" for tip in recommendations])}
</ul>
""", unsafe_allow_html=True)


# Proceed to Offset Button
# Proceed to Offset Button (Fixed to be one clickable button)
tab_result.markdown(
    """
    <div style='text-align: center; margin-top: 20px;'>
        <a href='https://www.ecosia.org/' target='_blank'>
            <button style='padding: 0.75em 1.5em; font-size: 1em; background-color: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer;'>
                🌿 Proceed to Offset Your Carbon Footprint
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)




# Footer and final script inclusion
with open("./style/footer.html", "r", encoding="utf-8") as footer:
    footer_html = f"""{footer.read()}"""
    st.markdown(footer_html, unsafe_allow_html=True)

script()
