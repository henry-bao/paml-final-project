import streamlit as st
import pandas as pd
from st_pages import add_page_title

add_page_title()

if "df" not in st.session_state:
    st.markdown("⚠️ Data not loaded. Please go to the [Home page](/) to load the data.")
    st.stop()

df = st.session_state.df
st.markdown("## Data Exploration")
st.write("### Data Summary")
st.write(df.describe())

st.write("### Data Types")
st.write(pd.DataFrame(df.dtypes, columns=["Data Type"]).T)

st.write("### Missing Values")
null_counts = df.isnull().sum()
non_null_counts = df.notnull().sum()
result = pd.DataFrame(
    {
        "Null Value Count": null_counts,
        "Non-Null Value Count": non_null_counts,
        "Percentage": round(null_counts / df.shape[0] * 100, 2),
    }
).T
st.write(result)


year_slider = st.slider(
    "Choose a year to see accident stats",
    min_value=int(df["Year"].min()),
    max_value=int(df["Year"].max()),
    value=int(df["Year"].max()),
)
total_accidents = df[df["Year"] == year_slider]["ID"].count()
average_severity = df[df["Year"] == year_slider]["Severity"].mean()
top_cities = df[df["Year"] == year_slider]["City"].value_counts().head(5)
st.markdown(f"#### Total accidents in {year_slider}: _{total_accidents}_")
st.markdown(f"#### Average severity in {year_slider}: _{average_severity:.2f}_")
st.markdown("#### Top 5 cities with the most accidents:")
for i, (city, count) in enumerate(top_cities.items()):
    st.markdown(
        f"{i + 1}. **{city}** - {count} accidents - {count / total_accidents * 100:.2f}% of total accidents - average severity: {df[df['City'] == city]['Severity'].mean():.2f}"
    )
