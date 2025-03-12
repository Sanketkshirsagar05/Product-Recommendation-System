import streamlit as st
import joblib
import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from PIL import Image

# Load the saved KNN model
model = joblib.load('knn_basics_model.pkl')

# Load the dataset (assuming df_final is available)
# df_final = pd.read_csv('path_to_your_final_dataset.csv')

# Streamlit app
st.set_page_config(page_title="Product Recommendation System", page_icon="🛍️", layout="centered")

# Add a title with an image
col1, col2 = st.columns([1, 3])
with col1:
    st.image("item_pic.jpg", width=100)  # Replace with your logo or image URL
with col2:
    st.title("Product Recommendation System 🛍️")

st.write("Welcome to the Product Recommendation System! Enter a product ID below to get the top 5 similar products.")

# Input product ID
product_id =st.text_input("**Enter the Product ID:**", placeholder="e.g., 0132793040")

if product_id:
    # Get the inner ID of the product
    try:
        product_inner_id = model.trainset.to_inner_iid(product_id)
    except ValueError:
        st.error("❌ Product ID not found in the dataset. Please enter a valid Product ID.")
        st.stop()

    # Get the nearest neighbors (similar products)
    neighbors = model.get_neighbors(product_inner_id, k=5)

    # Convert inner IDs back to product IDs
    similar_products = [model.trainset.to_raw_iid(inner_id) for inner_id in neighbors]

    # Display the similar products
    st.success("✅ Here are the top 5 similar products:")
    for i, product in enumerate(similar_products, 1):
        st.markdown(f"**{i}. {product}**")

# Add a footer with some additional information
st.markdown("---")
st.markdown(" About This App")
st.write("This app uses a KNN-based collaborative filtering model to recommend similar products. It helps users discover new products they might like!")

# Add some styling
st.markdown("""
<style>
    /* Main title styling */
    h1 {
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
    }

    /* Input field styling */
    .stTextInput>div>div>input {
        padding: 10px;
        border-radius: 8px;
        border: 2px solid #4CAF50;
        font-size: 1rem;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 28px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }

    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }

    /* Error message styling */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Add a footer
st.markdown('<div class="footer">© 2025 Product Recommendation System. All rights reserved.</div>', unsafe_allow_html=True)
