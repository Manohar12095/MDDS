import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load model and data
model = load_model("food_recognition_model.h5")
nutrition_df = pd.read_csv("data/nutrition_db.csv")
mapping_df = pd.read_csv("data/dish_mapping.csv")

def predict_food(image):
    img = tf.keras.utils.load_img(image, target_size=(128,128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Nutrition lookup
    food_id = mapping_df.loc[mapping_df["dish_label"] == predicted_class, "food_id"].values
    if len(food_id) > 0:
        row = nutrition_df.loc[nutrition_df["food_id"] == food_id[0]].iloc[0]
        nutrients = row.to_dict()
    else:
        nutrients = "Not found"

    return predicted_class, confidence, nutrients

# Streamlit UI
st.title("🍽️ Food Recognition App")
img_file = st.file_uploader("Upload food image", type=["jpg","png"])

if img_file:
    food, conf, nutri = predict_food(img_file)
    st.write("Predicted:", food, f"(Confidence: {conf:.2f}%)")
    st.write("Nutrition:", nutri)
