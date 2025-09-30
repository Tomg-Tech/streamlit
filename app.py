import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Explorer App", page_icon="📊")

st.title("📊 My Streamlit Data Explorer")
st.write("Upload a CSV file and explore your data!")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Show basic stats
    st.write("### Summary Statistics")
    st.write(df.describe())

    # Pick a column to plot
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        col = st.selectbox("Select column to plot", numeric_cols)
        fig, ax = plt.subplots()
        df[col].hist(bins=20, ax=ax)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    else:
        st.info("No numeric columns available to plot.")
else:
    st.info("Please upload a CSV file to begin.")
