import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load datasets
try:
    hotel_data = pd.read_excel('hotel_features_binary.xlsx')  # Replace with actual path
    restaurants_data = pd.read_csv('indian_restaurants.csv')  # Replace with actual path
    places_to_visit = pd.read_csv('Top Indian Places to Visit.csv')  # Replace with actual path
except Exception as e:
    st.error(f"Error loading data: {e}")

# Preprocessing the hotel data
X = hotel_data.drop(columns=['Hotel_Price', 'Hotel_Name'])  # Drop non-feature columns
y = hotel_data['Hotel_Price']  # Target variable (price)

# Define feature columns based on the hotel features
feature_columns = ['1 bed', '2 beds', 'Air conditioning', 'Airport shuttle', 'Bar', 'Beach access', 
                   'Breakfast', 'Cable TV', 'Elevator', 'Fireplace', 'Fitness center', 'Free parking', 
                   'Free Wi-Fi', 'Full-service laundry', 'Hot tub', 'Kid-friendly', 'Kitchen', 
                   'No air conditioning', 'No airport shuttle', 'Not pet-friendly', 'Paid parking', 
                   'Pet-friendly', 'Pool', 'Restaurant', 'Room service', 'Smoke-free', 'Spa', 'Wi-Fi']

# Encode the features
X_encoded = pd.get_dummies(X, columns=['city', 'City'] + feature_columns, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize models
decision_tree_model = DecisionTreeRegressor(random_state=42)
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train both models
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Evaluate both models
models = {'Decision Tree': decision_tree_model, 'Random Forest': random_forest_model}
metrics = {}

for model_name, model in models.items():
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics[model_name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

# Select the best model based on R2 score
best_model_name = max(metrics, key=lambda x: metrics[x]['R2'])
best_model = models[best_model_name]

# Model Evaluation Metrics Section
st.subheader("Model Evaluation Metrics")
for model_name, model_metrics in metrics.items():
    st.write(f"**{model_name}**")
    st.write(f"MAE: {model_metrics['MAE']:.2f}")
    st.write(f"MSE: {model_metrics['MSE']:.2f}")
    st.write(f"R²: {model_metrics['R2']:.2f}")
    st.write("---")

# Overall Interpretation of Model Performance
st.subheader("Model Comparison Interpretation")

best_model_mae = metrics[best_model_name]['MAE']
best_model_mse = metrics[best_model_name]['MSE']
best_model_r2 = metrics[best_model_name]['R2']

if best_model_name == 'Decision Tree':
    st.write("The Decision Tree model shows a balance between complexity and performance, but it may suffer from overfitting.")
else:
    st.write("The Random Forest model, being an ensemble method, usually offers more robustness against overfitting and handles non-linearities better.")

st.write(f"Overall, the best model is **{best_model_name}** with an R² score of {best_model_r2:.2f}, indicating that it explains {best_model_r2 * 100:.2f}% of the variance in hotel prices.")
st.write("The lower the MAE and MSE, the better the model performs, with the selected model showing acceptable error margins.")

# Optional: Visualize the model metrics
st.subheader("Model Performance Visualization")
metrics_df = pd.DataFrame(metrics)
metrics_df.plot(kind='bar', figsize=(10, 5), rot=0)
plt.title("Model Performance Comparison")
plt.ylabel("Error Metrics")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# Streamlit app interface
st.title("Hotel Price Prediction and City Guide")

st.header("Enter Hotel Information to Predict Price")
unique_cities = hotel_data['City'].unique()
city = st.selectbox("City", unique_cities)

# Hotel Rating
rating = st.slider("Hotel Rating", min_value=1.0, max_value=5.0, step=0.1)

# Prepare the input data for prediction
input_data = {
    'Hotel_Rating': [rating],
    'city_' + city: [1],  # Handle city as categorical
}

# Fill missing columns with 0s
input_df = pd.DataFrame(input_data)
input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)

# Predict Hotel Price
if st.button("Predict Hotel Price"):
    prediction = best_model.predict(input_df)
    predicted_price = prediction[0]

    # Display the predicted price
    st.write("Predicted Price: ₹{:.2f}".format(predicted_price))

    # Find hotels with the same rating across all cities
    all_hotels = hotel_data[hotel_data['Hotel_Rating'] == rating]

    # Display city name for context
    st.write(f"**Selected City:** {city}")

    # Get hotels within a ±10% price range
    price_range_min = predicted_price - (predicted_price * 0.10)
    price_range_max = predicted_price + (predicted_price * 0.10)

    # Get hotels within the predicted price range across all cities
    matching_hotels = all_hotels[
        (all_hotels['Hotel_Price'] >= price_range_min) & 
        (all_hotels['Hotel_Price'] <= price_range_max)
    ]

    if not matching_hotels.empty:
        st.write("Hotels with the selected rating in any city within the predicted price range:")
        for index, row in matching_hotels.iterrows():
            feature_description = ', '.join([feature for feature in feature_columns if row[feature] == 1])
            st.write(f"**Hotel:** {row['Hotel_Name']}, **City:** {row['City']}, **Price:** ₹{row['Hotel_Price']:.2f}, **Features:** {feature_description}")

        # Find other cities with the same rating
        other_cities = hotel_data[hotel_data['Hotel_Rating'] == rating]['City'].unique()
        other_cities = [c for c in other_cities if c != city]  # Exclude the selected city

        if other_cities:
            st.write("Cities with hotels having the same rating:")
            for other_city in other_cities:
                st.write(f"- {other_city}")

                # Get the best hotels from this city (highest price for the rating)
                best_hotels = hotel_data[(hotel_data['City'] == other_city) & (hotel_data['Hotel_Rating'] == rating)]
                if not best_hotels.empty:
                    best_hotels = best_hotels.sort_values(by='Hotel_Price', ascending=False).head(3)  # Get top 3 hotels
                    st.write(f"**Best Hotels in {other_city}:**")
                    for index, best_row in best_hotels.iterrows():
                        feature_description = ', '.join([feature for feature in feature_columns if best_row[feature] == 1])
                        st.write(f"  - **Hotel:** {best_row['Hotel_Name']}, **Price:** ₹{best_row['Hotel_Price']:.2f}, **Features:** {feature_description}")
                else:
                    st.write(f"No hotels found in {other_city} with rating {rating}.")
        else:
            st.write("No other cities found with hotels having the same rating.")
    else:
        st.write("No hotels found within the predicted price range.")

# Cuisine Type Checkboxes
st.subheader("Select Cuisine Types")
cuisines = {
    'south_indian_or_not': 'South Indian',
    'north_indian_or_not': 'North Indian',
    'fast_food_or_not': 'Fast Food',
    'street_food': 'Street Food',
    'biryani_or_not': 'Biryani',
    'bakery_or_not': 'Bakery'
}

# Initialize input data for cuisine
input_data = {
    'Hotel_Rating': [rating],
    'city_' + city: [1],
}

# Track selected cuisines
selected_cuisines = []
for cuisine_key, cuisine_label in cuisines.items():
    if st.checkbox(cuisine_label, value=False):
        input_data[cuisine_key] = [1]  # User selected this cuisine
        selected_cuisines.append(cuisine_key)
    else:
        input_data[cuisine_key] = [0]  # User did not select this cuisine

# Prepare the input DataFrame for prediction
input_df = pd.DataFrame(input_data)
input_df = input_df.reindex(columns=X_encoded.columns, fill_value=0)

# Nearby Restaurants Section
st.header("Nearby Restaurants")

# Filter restaurants based on the selected city
filtered_restaurants = restaurants_data[restaurants_data['City'] == city]

# If no cuisines are selected, show all restaurants in the city
if selected_cuisines:
    # Filter restaurants based on selected cuisines
    filtered_restaurants = filtered_restaurants[filtered_restaurants[selected_cuisines].any(axis=1)]

# Add a slider to filter by restaurant rating
min_rating = st.slider("Minimum Restaurant Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# Further filter restaurants based on the selected rating
filtered_restaurants = filtered_restaurants[filtered_restaurants['rating'] >= min_rating]

# Handle potential NaN values in average_price
if 'average_price' in filtered_restaurants.columns:
    filtered_restaurants['average_price'].fillna(0, inplace=True)

    # Ensure there are valid prices for the slider
    if not filtered_restaurants['average_price'].empty:
        min_price = int(filtered_restaurants['average_price'].min())
        max_price = int(filtered_restaurants['average_price'].max())

        # Add a slider to filter by restaurant price (using integers)
        price_range = st.slider(
            "Price Range",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )

        # Filter restaurants based on the selected price range
        filtered_restaurants = filtered_restaurants[
            (filtered_restaurants['average_price'] >= price_range[0]) & 
            (filtered_restaurants['average_price'] <= price_range[1])
        ]

# Create a new column that combines the cuisines
filtered_restaurants['cuisines'] = filtered_restaurants[selected_cuisines].apply(lambda x: ', '.join(x.index[x.astype(bool)]), axis=1)

if not filtered_restaurants.empty:
    st.write(f"**Found {len(filtered_restaurants)} restaurants in {city}:**")

    # Create a DataFrame for filtered restaurants
    restaurant_df = filtered_restaurants[['restaurant_name', 'rating', 'average_price']].copy()

    # Rename columns for display
    restaurant_df = restaurant_df.rename(columns={
        'restaurant_name': 'Restaurant Name',
        'rating': 'Rating',
        'average_price': 'Price (₹)',
        'cuisines': 'Cuisines'
    })

    # Display the restaurant DataFrame as a table
    st.table(restaurant_df)
else:
    st.write("No restaurants found matching your criteria.")

st.header("Top Places to Visit")

# Filter by place type (assuming a 'Category' column exists)
place_types = places_to_visit['Type'].unique()
selected_place_type = st.selectbox("Select Place Type", ["All"] + list(place_types))

# Filter places based on the selected city and place type
if selected_place_type == "All":
    filtered_places = places_to_visit[places_to_visit['City'] == city]
else:
    filtered_places = places_to_visit[(places_to_visit['City'] == city) & 
                                       (places_to_visit['Type'] == selected_place_type)]

# Top Places to Visit in Table Format
if not filtered_places.empty:
    st.write(f"Top Places to Visit in {city}:")
    st.table(filtered_places[['Name', 'Significance', 'Entrance Fee in INR']].rename(columns={
        'Name': 'Place Name',
        'Significance': 'Significance',
        'Entrance Fee in INR': 'Entrance Fee (₹)'
    }))
else:
    st.write("No tourist places found for this city.")
