import streamlit as st
from st_pages import Page, show_pages, add_page_title
import pandas as pd

show_pages(
    [
        Page("app.py", "Home", "🏠"),
        Page("pages/data_explore.py", "Data Exploration", "🗺️"),
        Page("pages/data_visualization.py", "Data Visualization", "📈"),
        Page("pages/model_prediction.py", "Model Prediction", "🚗"),
    ]
)

add_page_title()
st.markdown("# Welcome to the US Car Accident Insights Dashboard")


# Load data
def load_csv_with_progress(file_path):
    msg = st.markdown(f"🧪 Loading data from {file_path}...")
    my_bar = st.progress(0)
    total_rows = sum(1 for _ in open(file_path)) - 1
    processed_rows = 0
    chunk_size = 200

    data = pd.DataFrame()

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        data = pd.concat([data, chunk], ignore_index=True)
        processed_rows += len(chunk)
        my_bar.progress(processed_rows / total_rows)

    msg.empty()
    my_bar.empty()

    return data


df = None

if "df" not in st.session_state:
    df = load_csv_with_progress("./data/US_Accidents_March23_random_sample.csv")
    st.session_state["df"] = df
else:
    df = st.session_state["df"]

if df is not None:
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["Year"] = df["Start_Time"].dt.year

    st.markdown("## How to Use This App")
    st.markdown(
        """
        - Navigate to the **Data Exploration** page to view detailed visualizations by state or city.
        - Use the **Model Training** page to estimate accident severity based on various factors.
        - Interact with the charts and maps for deeper insights.
    """
    )

    st.markdown("## Did You Know?")
    st.markdown(
        """
        - More than 38,000 people die every year in crashes on U.S. roadways.
        - The U.S. traffic fatality rate is 12.4 deaths per 100,000 inhabitants.
        - Seat belts reduce the risk of death by 45% for drivers and front-seat passengers.
    """
    )

    st.markdown("## Quick Data Preview")
    st.write(df.head(15))

    st.markdown("## Acknowledgements")
    st.markdown(
        """
        - [Kaggle](https://www.kaggle.com/sobhanmoosavi/us-accidents) for the dataset.
        - [Streamlit](https://streamlit.io) for the amazing app framework.
    """
    )
