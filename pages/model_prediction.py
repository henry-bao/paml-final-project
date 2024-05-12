import streamlit as st
from st_pages import add_page_title
import joblib
import numpy as np
import pandas as pd
import json

from models.decision_tree import DecisionTree
from sklearn.ensemble import RandomForestClassifier

add_page_title()

if "df" not in st.session_state:
    st.markdown("⚠️ Data not loaded. Please go to the [Home page](/) to load the data.")
    st.stop()("⚠️ Data not loaded. Please go to the Home page to load the data.")

rf_model = joblib.load("models/random_forest_model.pkl")
with open("data/state_city_county.json", "r") as json_file:
    loc_data = json.load(json_file)

st.markdown(
    "This section allows you to predict accident severity by selecting a model and entering relevant data."
)

model_choice = st.selectbox(
    "Select a Model", ["Decision Trees", "Random Forest", "K-Nearest Neighbors"]
)

st.markdown("### Select Location")

col1, col2, col3 = st.columns(3)
with col1:
    state = st.selectbox(
        "Choose a State:",
        list(loc_data.keys()),
        key="state",
        index=list(loc_data.keys()).index("NY"),
    )

if state:
    cities = list(loc_data[state].keys())
    with col2:
        city = st.selectbox(
            "Choose a City:", cities, key="city", index=cities.index("New York")
        )
    if city:
        counties = loc_data[state][city]
        with col3:
            county = st.selectbox("Choose a County:", counties, key="county")

# Time
st.markdown("### Select Time")
col1, col2, col3 = st.columns(3)
with col1:
    month_dict = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    month = st.selectbox("Month", list(month_dict.keys()), key="month")
    month = month_dict[month]
with col2:
    weekday_dict = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
    }
    weekday = st.selectbox("Day of the Week", list(weekday_dict.keys()), key="weekday")
    weekday = weekday_dict[weekday]

with col3:
    hour = st.selectbox("Hour", list(range(0, 24)), key="hour")

# Environment
st.markdown("### Select Driving Environment")

col1, col2 = st.columns(2)
with col1:
    weather = st.selectbox(
        "Weather Condition",
        ["Clear", "Cloud", "Rain", "Heavy_Rain", "Snow", "Heavy_Snow", "Fog"],
        key="weather",
    )
with col2:
    wind_direction = st.selectbox(
        "Wind Direction",
        ["S", "NE", "E", "CALM", "N", "SE", "VAR", "SW", "W", "NW"],
        key="wind_direction",
    )

col1, col2, col3 = st.columns(3)
with col1:
    wind_speed = st.slider("Wind Speed(mph)", 0, 175, 10, 5, key="wind_speed")
with col2:
    temperature = st.slider("Temperature(F)", -35, 160, 65, 5, key="temperature")
with col3:
    humidity = st.slider("Humidity(%)", 0, 100, 50, 5, key="humidity")

col1, col2, col3 = st.columns(3)
with col1:
    visibility = st.slider("Visibility(mi)", 0.0, 10.0, 5.0, 0.5, key="visibility")
with col2:
    pressure = st.slider("Pressure(in)", 15.0, 55.0, 29.5, 0.5, key="pressure")
with col3:
    precipitation = st.slider(
        "Precipitation(in)", 0.0, 10.0, 0.0, 0.5, key="precipitation"
    )

route_features = st.multiselect(
    "Select Route Features",
    [
        "Amenity",
        "Bump",
        "Crossing",
        "Give_Way",
        "Junction",
        "No_Exit",
        "Railway",
        "Roundabout",
        "Station",
        "Stop",
        "Traffic_Calming",
        "Traffic_Signal",
        "Turning_Loop",
    ],
    key="route_features",
)

predict_button = st.button("Predict Severity")


def has_route_feature(feature):
    return feature in route_features


def is_weather(w):
    return weather == w


# Button to perform prediction

if predict_button:
    time_of_day = "Day" if 6 <= hour < 18 else "Night"
    features = pd.DataFrame(
        {
            "City": [city],
            "County": [county],
            "State": [state],
            "Temperature(F)": [temperature],
            "Humidity(%)": [humidity],
            "Pressure(in)": [pressure],
            "Visibility(mi)": [visibility],
            "Wind_Direction": [wind_direction],
            "Wind_Speed(mph)": [wind_speed],
            "Precipitation(in)": [precipitation],
            "Amenity": [has_route_feature("Amenity")],
            "Bump": [has_route_feature("Bump")],
            "Crossing": [has_route_feature("Crossing")],
            "Give_Way": [has_route_feature("Give_Way")],
            "Junction": [has_route_feature("Junction")],
            "No_Exit": [has_route_feature("No_Exit")],
            "Railway": [has_route_feature("Railway")],
            "Roundabout": [has_route_feature("Roundabout")],
            "Station": [has_route_feature("Station")],
            "Stop": [has_route_feature("Stop")],
            "Traffic_Calming": [has_route_feature("Traffic_Calming")],
            "Traffic_Signal": [has_route_feature("Traffic_Signal")],
            "Sunrise_Sunset": [time_of_day],
            "Civil_Twilight": [time_of_day],
            "Nautical_Twilight": [time_of_day],
            "Astronomical_Twilight": [time_of_day],
            "Clear": [is_weather("Clear")],
            "Cloud": [is_weather("Cloud")],
            "Rain": [is_weather("Rain")],
            "Heavy_Rain": [is_weather("Heavy_Rain")],
            "Snow": [is_weather("Snow")],
            "Heavy_Snow": [is_weather("Heavy_Snow")],
            "Fog": [is_weather("Fog")],
            "Month": [month],
            "Weekday": [weekday],
            "Hour": [hour],
        },
        index=[0],  # specify index for the DataFrame
    )

    features_dummy = pd.get_dummies(pd.DataFrame(features, columns=features.columns))

    prediction = rf_model.predict(features_dummy)

    # This part simulates the prediction
    st.success(f"Model {model_choice} would predict severity here based on inputs.")
    # Optionally, add a placeholder for where actual prediction results would go
    st.write("This is where the prediction result would appear.")
    st.write(f"Predicted Severity: {prediction}")
    # # Dummy performance metrics (These would be dynamically calculated with actual model predictions)
    # precision = 0.75  # Example precision value
    # recall = 0.65  # Example recall value
    # f1 = 0.70  # Example F1 score

    # # Displaying the metrics
    # st.write(f"### Performance Metrics for {model_choice} Model")
    # st.write(f"**Precision:** {precision}")
    # st.write(f"**Recall:** {recall}")
    # st.write(f"**F1 Score:** {f1}")

    # If you had a real prediction, you could display it like so:
    # st.write(f"The predicted severity is: {predicted_severity}")
