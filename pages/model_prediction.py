import streamlit as st
from st_pages import add_page_title

add_page_title()

if "df" not in st.session_state:
    st.markdown(f"⚠️ Data not loaded. Please go to the [Home page](/) to load the data.")
    st.stop()("⚠️ Data not loaded. Please go to the Home page to load the data.")

st.markdown(
    "This section allows you to predict accident severity by selecting a model and entering relevant data."
)

model_choice = st.selectbox(
    "Select a model", ["Decision Trees", "Random Forest", "K-Nearest Neighbors"]
)

# Input fields for features
feature1 = st.number_input(
    "Feature 1 (e.g., Speed):", min_value=0.0, max_value=100.0, value=50.0, step=0.1
)
feature2 = st.number_input(
    "Feature 2 (e.g., Weather Condition Index):", min_value=0, max_value=10, value=5
)
feature3 = st.number_input(
    "Feature 3 (e.g., Hour of the Day):", min_value=0, max_value=23, value=12
)

# Button to perform prediction
predict_button = st.button("Predict Severity")

if predict_button:
    # This part simulates the prediction
    st.success(f"Model {model_choice} would predict severity here based on inputs.")
    st.info(f"Feature 1: {feature1}, Feature 2: {feature2}, Feature 3: {feature3}")
    # Optionally, add a placeholder for where actual prediction results would go
    st.write("This is where the prediction result would appear.")
    # Dummy performance metrics (These would be dynamically calculated with actual model predictions)
    precision = 0.75  # Example precision value
    recall = 0.65  # Example recall value
    f1 = 0.70  # Example F1 score

    # Displaying the metrics
    st.write(f"### Performance Metrics for {model_choice} Model")
    st.write(f"**Precision:** {precision}")
    st.write(f"**Recall:** {recall}")
    st.write(f"**F1 Score:** {f1}")

    # If you had a real prediction, you could display it like so:
    # st.write(f"The predicted severity is: {predicted_severity}")
