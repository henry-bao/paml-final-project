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

top_cities_with_state = df[df["Year"] == year_slider].groupby(["City", "State"]).size()


st.markdown(f"#### Total accidents in {year_slider}: _{total_accidents}_")
st.markdown(f"#### Average severity in {year_slider}: _{average_severity:.2f}_")
st.info("Top 5 **cities** with the **most** accidents:")
# city, state - accidents - % of total accidents - average severity
for i, (city, state) in enumerate(
    top_cities_with_state.sort_values(ascending=False).head(5).items()
):
    city, state = city
    city_accidents = df[
        (df["Year"] == year_slider) & (df["City"] == city) & (df["State"] == state)
    ].shape[0]
    city_severity = df[
        (df["Year"] == year_slider) & (df["City"] == city) & (df["State"] == state)
    ]["Severity"].mean()
    st.markdown(
        f"{i + 1}. **{city}, {state}** - {city_accidents} accidents - {city_accidents / total_accidents * 100:.2f}% of total accidents - Average Severity: {city_severity:.2f}"
    )


st.info("Top 5 **states** with the **most** accidents:")
for i, (state, count) in enumerate(
    df[df["Year"] == year_slider]["State"].value_counts().head(5).items()
):
    st.markdown(
        f"{i + 1}. **{state}** - {count} accidents - {count / total_accidents * 100:.2f}% of total accidents - Average Severity: {df[(df['Year'] == year_slider) & (df['State'] == state)]['Severity'].mean():.2f}"
    )
