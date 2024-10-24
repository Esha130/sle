import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load datasets
hotel_data = pd.read_csv('google_hotel_data_clean_v2.csv')  # Replace with actual path
restaurants_data = pd.read_csv('indian_restaurants_data.csv')  # Replace with actual path
places_to_visit = pd.read_csv('Top Indian Places to Visit.csv')  # Replace with actual path

# Preprocessing the hotel data
X = hotel_data.drop(columns=['Hotel_Name', 'Hotel_Price'])  # Features
y = hotel_data['Hotel_Price']  # Target variable (price)
X_encoded = pd.get_dummies(X, columns=['City', 'Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 
                                       'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9'], drop_first=True)

# Train the model (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_encoded, y)

# Streamlit app interface
st.title("Hotel Price Prediction and City Guide")

# Hotel Prediction Section
st.header("Enter Hotel Information to Predict Price")
city = st.selectbox("City", hotel_data['City'].unique())
rating = st.slider("Hotel Rating", min_value=1.0, max_value=5.0, step=0.1)
features = {}
for i in range(1, 10):
    features[f'Feature_{i}'] = st.selectbox(f'Feature {i}', hotel_data[f'Feature_{i}'].unique())

# Prepare the input data for prediction
input_data = {
    'Hotel_Rating': [rating],
    'City_' + city: [1],
}

for i in range(1, 10):
    input_data[f'Feature_{i}_{features[f"Feature_{i}"]}'] = [1]

# Fill missing columns with 0s
input_df = pd.DataFrame(input_data)
input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)

# Predict Hotel Price
if st.button("Predict Hotel Price"):
    prediction = rf_model.predict(input_df)
    st.write(f"The predicted price for this hotel is: ₹{prediction[0]:.2f}")

# Nearby Restaurants Section
st.header("Nearby Restaurants")
filtered_restaurants = restaurants_data[restaurants_data['City'] == city]
if not filtered_restaurants.empty:
    st.write(filtered_restaurants[['Restaurant_Name', 'Cuisine', 'Rating']])
else:
    st.write("No restaurants found for this city.")

# Top Places to Visit Section
st.header("Top Places to Visit")
filtered_places = places_to_visit[places_to_visit['City'] == city]
if not filtered_places.empty:
    st.write(filtered_places[['Place_Name', 'Description']])
else:
    st.write("No tourist places found for this city.")

# Show the hotel dataset if needed
if st.checkbox("Show hotel dataset"):
    st.write(hotel_data)

