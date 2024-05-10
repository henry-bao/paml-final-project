import streamlit as st
from st_pages import Page, show_pages, add_page_title

show_pages(
    [
        Page("app.py", "Home", ":house:"),
        Page("pages/test.py", "Page 2", ":books:"),
    ]
)

add_page_title()

st.markdown("hello world")
