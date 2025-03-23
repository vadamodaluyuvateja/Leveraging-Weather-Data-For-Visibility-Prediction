import streamlit as st
import os
import random
import requests
import csv
import folium
import pandas as pd
from geopy.geocoders import OpenCage
from geopy.distance import geodesic
from datetime import datetime, timedelta
import shutil

# Import your existing prediction modules
from prediction_Validation_Insertion import pred_validation
from predictFromModel import prediction

def get_weather_data(lat, lon, api_key):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            current_weather = data['current']
            dry_bulb_temp_c = current_weather['temp_c']
            dew_point_c = current_weather['dewpoint_c']
            relative_humidity = current_weather['humidity']

            # Approximate Wet Bulb Temperature (°C) using empirical formula
            wet_bulb_temp_c = dew_point_c + 0.36 * (dry_bulb_temp_c - dew_point_c)

            weather_data = {
                "DRYBULBTEMPF": round((dry_bulb_temp_c * 9 / 5) + 32, 2),
                "WETBULBTEMPF": round((wet_bulb_temp_c * 9 / 5) + 32, 2),
                "DewPointTempF": round((dew_point_c * 9 / 5) + 32, 2),
                "RelativeHumidity": relative_humidity,
                "WindSpeed": round(current_weather['wind_kph'] * 0.621371, 2),
                "WindDirection": current_weather['wind_degree'],
                "StationPressure": round(current_weather['pressure_mb'] * 0.02953, 2),
                "SeaLevelPressure": round(current_weather['pressure_mb'] * 0.02953, 2),
                "Precip": round(current_weather['precip_mm'] / 25.4, 2),
            }
            return weather_data
        except KeyError as e:
            return {"error": f"Missing data: {e}"}
    else:
        return {"error": response.text}

def generate_random_points(start, end, num_points=20):
    points = []
    for _ in range(num_points):
        t = random.uniform(0, 1)
        lat = start[0] + t * (end[0] - start[0])
        lon = start[1] + t * (end[1] - start[1])
        points.append((lat, lon))
    return points

def get_coordinates(location, api_key):
    geolocator = OpenCage(api_key=api_key)
    try:
        location_info = geolocator.geocode(location)
        if location_info:
            return location_info.latitude, location_info.longitude
        else:
            st.error(f"Error: {location} not found.")
            return None
    except Exception as e:
        st.error(f"Error fetching coordinates for {location}: {e}")
        return None

def calculate_distance(start_coords, end_coords):
    # Calculate distance in kilometers
    distance_km = geodesic(start_coords, end_coords).kilometers
    # Convert to miles
    distance_miles = distance_km * 0.621371
    return distance_km, distance_miles

def create_map(start_coords, end_coords, points, distance_miles, map_file_name="map.html"):
    m = folium.Map(location=[(start_coords[0] + end_coords[0]) / 2, 
                              (start_coords[1] + end_coords[1]) / 2], 
                   zoom_start=5)
    
    # Add markers with distance information
    start_popup = f"Start Location"
    end_popup = f"End Location<br>Distance: {distance_miles:.2f} miles"
    
    folium.Marker(location=start_coords, popup=start_popup).add_to(m)
    folium.Marker(location=end_coords, popup=end_popup).add_to(m)
    
    for point in points:
        folium.Marker(location=point, icon=folium.Icon(color="blue")).add_to(m)
    
    folium.PolyLine([start_coords, end_coords], color="green", weight=2.5, opacity=1).add_to(m)
    
    # Create maps directory if it doesn't exist
    os.makedirs("maps", exist_ok=True)
    m.save(os.path.join("maps", map_file_name))
    return os.path.join("maps", map_file_name)

def main():
    st.title("Weather Data Generation and Prediction App")

    # Sidebar for navigation - removed Model Training option
    menu = ["Weather Data Generation", "Prediction", "View Predictions"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # OpenCage and WeatherAPI keys
    api_key_opencage = "fae2d6415b13473eb2ddab20d7507c43"
    api_key_weather = "fcefb1a789364bfbb5854916251203"

    if choice == "Weather Data Generation":
        st.subheader("Generate Weather Data Along Route")
        
        # Input locations
        start_location = st.text_input("Start Location")
        end_location = st.text_input("End Location")
        
        # Add input for custom directory
        custom_directory = st.text_input("Custom Directory Name (optional)")
        
        # Checkbox for automatic prediction
        auto_predict = st.checkbox("Automatically Generate Prediction After Data Generation")
        
        if st.button("Generate Weather Data"):
            # Ensure Prediction_Batch_files directory exists
            os.makedirs("Prediction_Batch_files", exist_ok=True)
            
            # Clear previous CSV files in Prediction_Batch_files directory
            for file in os.listdir("Prediction_Batch_files"):
                if file.endswith(".csv"):
                    os.remove(os.path.join("Prediction_Batch_files", file))
            
            # Get coordinates
            start_coords = get_coordinates(start_location, api_key_opencage)
            end_coords = get_coordinates(end_location, api_key_opencage)

            if start_coords and end_coords:
                # Calculate distance
                distance_km, distance_miles = calculate_distance(start_coords, end_coords)
                
                # Display distance
                st.subheader("Route Information")
                st.write(f"Distance: {distance_miles:.2f} miles")
                
                # Generate random points
                points = generate_random_points(start_coords, end_coords, 20)
                
                # Collect weather data
                weather_data_list = []
                base_time = datetime.now()

                # Progress bar
                progress_bar = st.progress(0)
                for idx, (lat, lon) in enumerate(points):
                    st.write(f"Fetching weather data for Point {idx + 1}...")
                    weather_data = get_weather_data(lat, lon, api_key_weather)
                    
                    if "error" not in weather_data:
                        weather_data["DATE"] = (base_time + timedelta(hours=idx)).strftime("%d-%m-%Y %H:%M")
                        weather_data_list.append(weather_data)
                    
                    # Update progress bar
                    progress_bar.progress((idx + 1) / len(points))

                # Generate CSV
                now = datetime.now()
                formatted_date = now.strftime('%d%m%Y_%H%M%S')
                
                # Use custom directory if provided
                target_directory = custom_directory if custom_directory else "imp"
                os.makedirs(target_directory, exist_ok=True)
                
                # CSV filename
                csv_file = f"visibility_{formatted_date}.csv"
                csv_path = os.path.join(target_directory, csv_file)

                # Save to CSV
                df = pd.DataFrame(weather_data_list)
                df.to_csv(csv_path, index=False)
                st.success(f"Weather data saved to {csv_path}")

                # Copy CSV to Prediction_Batch_files for automatic prediction
                prediction_batch_path = os.path.join("Prediction_Batch_files", csv_file)
                shutil.copy(csv_path, prediction_batch_path)
                st.success(f"Copied to {prediction_batch_path}")

                # Create map with distance information
                map_file = create_map(start_coords, end_coords, points, distance_miles,
                                      f"{start_location}to{end_location}_map.html")
                st.success(f"Map saved to {map_file}")
                
                # Display map
                with open(map_file, 'r') as f:
                    st.components.v1.html(f.read(), height=600)
                
                # Automatic Prediction if checkbox is checked
                if auto_predict:
                    try:
                        # Validation
                        pred_val = pred_validation("Prediction_Batch_files")
                        pred_val.prediction_validation()

                        # Prediction
                        pred = prediction("Prediction_Batch_files")
                        output_path = pred.predictionFromModel()

                        st.success(f"Automatic Prediction Generated at {output_path}")
                        
                        # Display prediction averages
                        if os.path.exists("Prediction_Output_File/Predictions.csv"):
                            pred_df = pd.read_csv("Prediction_Output_File/Predictions.csv")
                            
                            st.subheader("Prediction Averages")
                            avg_data = {}
                            numeric_columns = pred_df.select_dtypes(include=['float64', 'int64']).columns
                            
                            for col in numeric_columns:
                                avg_data[col] = pred_df[col].mean()
                            
                            avg_df = pd.DataFrame([avg_data])
                            st.write(avg_df)
                            
                            # Add the specific visibility summary message
                            if 'Visibility' in pred_df.columns:
                                avg_visibility = pred_df['Visibility'].mean()
                                st.success(f"{avg_visibility:.1f}-miles is the avg visibility distance from {start_location} to {end_location}")
                            else:
                                # Try to determine the visibility column name
                                possible_visibility_cols = [col for col in pred_df.columns if 'vis' in col.lower()]
                                if possible_visibility_cols:
                                    avg_visibility = pred_df[possible_visibility_cols[0]].mean()
                                    st.success(f"{avg_visibility:.1f}-miles is the avg visibility distance from {start_location} to {end_location}")
                    except Exception as e:
                        st.error(f"Automatic Prediction failed: {e}")

    elif choice == "Prediction":
        st.subheader("Generate Predictions")
        prediction_path = st.text_input("Enter Prediction Data Path")
        start_location = st.text_input("Start Location (for summary)")
        end_location = st.text_input("End Location (for summary)")
        
        if st.button("Generate Prediction"):
            try:
                # Validation
                pred_val = pred_validation(prediction_path)
                pred_val.prediction_validation()

                # Prediction
                pred = prediction(prediction_path)
                output_path = pred.predictionFromModel()

                st.success(f"Prediction File created at {output_path}")
                
                # Display prediction averages
                if os.path.exists("Prediction_Output_File/Predictions.csv"):
                    pred_df = pd.read_csv("Prediction_Output_File/Predictions.csv")
                    
                    st.subheader("Prediction Averages")
                    avg_data = {}
                    numeric_columns = pred_df.select_dtypes(include=['float64', 'int64']).columns
                    
                    for col in numeric_columns:
                        avg_data[col] = pred_df[col].mean()
                    
                    avg_df = pd.DataFrame([avg_data])
                    st.write(avg_df)
                    
                    # Add the specific visibility summary message
                    if 'Visibility' in pred_df.columns:
                        avg_visibility = pred_df['Visibility'].mean()
                        st.success(f"{avg_visibility:.1f}-miles is the avg visibility distance from {start_location} to {end_location}")
                    else:
                        # Try to determine the visibility column name
                        possible_visibility_cols = [col for col in pred_df.columns if 'vis' in col.lower()]
                        if possible_visibility_cols:
                            avg_visibility = pred_df[possible_visibility_cols[0]].mean()
                            st.success(f"{avg_visibility:.1f}-miles is the avg visibility distance from {start_location} to {end_location}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif choice == "View Predictions":
        st.subheader("Prediction Results")
        start_location = st.text_input("Start Location (for summary)")
        end_location = st.text_input("End Location (for summary)")
        
        predictions_file = "Prediction_Output_File/Predictions.csv"
        if os.path.exists(predictions_file):
            df = pd.read_csv(predictions_file)
            st.dataframe(df)
            
            st.subheader("Prediction Averages")
            avg_data = {}
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            
            for col in numeric_columns:
                avg_data[col] = df[col].mean()
            
            avg_df = pd.DataFrame([avg_data])
            st.write(avg_df)
            
            # Add the specific visibility summary message
            if 'Visibility' in df.columns:
                avg_visibility = df['Visibility'].mean()
                st.success(f"{avg_visibility:.1f}-miles is the avg visibility distance from {start_location} to {end_location}")
            else:
                # Try to determine the visibility column name
                possible_visibility_cols = [col for col in df.columns if 'vis' in col.lower()]
                if possible_visibility_cols:
                    avg_visibility = df[possible_visibility_cols[0]].mean()
                    st.success(f"{avg_visibility:.1f}-miles is the avg visibility distance from {start_location} to {end_location}")
        else:
            st.warning("No prediction file found. Please run a prediction first.")

if __name__ == "__main__":
    main()
