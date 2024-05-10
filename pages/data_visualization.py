import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from st_pages import add_page_title

add_page_title()

if "df" not in st.session_state:
    st.markdown(f"⚠️ Data not loaded. Please go to the [Home page](/) to load the data.")
    st.stop()


df = st.session_state["df"]
df["Severity4"] = 0
df.loc[df["Severity"] == 4, "Severity4"] = 1
severity4 = df[df["Severity4"] == 1]

st.markdown("## Accidents Over Time")
accidents_by_year = df["Year"].value_counts().sort_index()
fig = px.line(
    x=accidents_by_year.index,
    y=accidents_by_year.values,
    labels={"x": "Year", "y": "Number of Accidents"},
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("## Accidents by the Time of Day")
df["Hour"] = df["Start_Time"].dt.hour
fig = px.bar(
    x=df["Hour"].value_counts().index,
    y=df["Hour"].value_counts().values,
    labels={"x": "Hour", "y": "Number of Accidents"},
)
st.plotly_chart(fig, use_container_width=True)

st.write("### Top 10 Weather Condition for Accidents")
weather_count = df["Weather_Condition"].value_counts().head(10)
fig = px.bar(
    x=weather_count.index,
    y=weather_count.values,
    labels={"x": "Weather Condition", "y": "Number of Accidents"},
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("## Accidents by State")
accident_counts = df["State"].value_counts().reset_index()
accident_counts.columns = ["state", "counts"]
fig = px.choropleth(
    accident_counts,
    locations="state",
    locationmode="USA-states",
    color="counts",
    scope="usa",
    color_continuous_scale="OrRd",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("## Map of Accidents with Severity Level 4")
plt.figure(figsize=(15, 10))
plt.plot(
    "Start_Lng",
    "Start_Lat",
    data=df,
    linestyle="",
    marker="o",
    markersize=1.5,
    color="teal",
    alpha=0.2,
    label="All Accidents",
)
plt.plot(
    "Start_Lng",
    "Start_Lat",
    data=severity4,
    linestyle="",
    marker="o",
    markersize=1.5,
    color="red",
    alpha=0.5,
    label="Accidents with Serverity Level 4",
)
plt.legend(markerscale=8)
plt.xlabel("Longitude", size=12, labelpad=3)
plt.ylabel("Latitude", size=12, labelpad=3)

st.pyplot(plt)
