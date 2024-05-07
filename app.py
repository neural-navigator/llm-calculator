import streamlit as st
from memory_calculator import ModelUtils


def my_fun(repo_name, access_token=None):
    model_utils = ModelUtils(repo_name, access_token)
    inference_cost = model_utils.get_inference_memory()
    return inference_cost


# Set page configuration
st.set_page_config(page_title="LLM Memory Calculator!", layout="wide")

st.title("LLM Calculator")

container_width = "50%"
margin = "auto"

# Create columns with negative left/right margins
col1, col2, col3 = st.columns([0.25, 0.5, 0.25])

# Apply negative margin style to center column (col2)
with col2:
    repo_name = st.text_input("Please enter the repo name: ")
    access_token = st.text_input("Please enter the huggingface access token: ")

    if st.button("Submit"):
        res = my_fun(repo_name, access_token)
        res = round(res, 2)
        st.text(f"{res} GB")
