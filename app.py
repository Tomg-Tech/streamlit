import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI  # or the correct import for your version

st.set_page_config(page_title="Data Explorer App + AI", page_icon="🤖")

st.title("🤖 My Streamlit AI Data Explorer")
st.write("Upload a CSV file, explore data, and chat with an AI model!")

# Load API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize model
model_name = "gpt-4o-mini"  # for example
creative_model = ChatOpenAI(
    model=model_name,
    api_key=api_key,
    temperature=0.7
)

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
else:
    st.info("No file uploaded — showing sample dataset.")
    df = pd.read_csv("example_data.csv")
    st.write("### Sample Data Preview", df.head())

# Show basic stats
st.write("### Summary Statistics")
st.write(df.describe())

# Plot numeric column
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
if len(numeric_cols) > 0:
    col = st.selectbox("Select column to plot", numeric_cols)
    fig, ax = plt.subplots()
    df[col].hist(bins=20, ax=ax)
    ax.set_title(f"Histogram of {col}")
    st.pyplot(fig)

# AI Q&A Section
st.write("### 💬 Ask the AI about your data")
user_q = st.text_area("Type a question about the dataset:")
if st.button("Ask AI") and user_q:
    response = creative_model.invoke(user_q)
    st.success(response.content)
