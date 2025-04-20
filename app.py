import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2
import os

# Define the model architecture
class HDBNet(nn.Module):
    def __init__(self, input_dim):
        super(HDBNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model_state = torch.load("hdb_model.pt", map_location=torch.device("cpu"))

# Dummy input to extract expected input dimension
dummy_input = np.zeros((1, 348))  # 348 = number of features from training
model = HDBNet(input_dim=dummy_input.shape[1])
model.load_state_dict(model_state)
model.eval()
# Load default autofill values by town
default_df = pd.read_csv("df_merged.csv")
address_data = pd.read_csv("address_data_full.csv")

# Compute mean values per town for autofill
town_defaults = default_df.groupby('town')[[
    'cpi', 'Address Lat', 'Address Long',
    'Nearest MRT Distance', 'Nearest Mall Distance', 'Nearest NPC Distance'
]].mean().round(3)
# Load dataset for autofill + dropdown valu
df = pd.read_csv("df_merged.csv")

# Load dropdown options
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG',
                'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS',
                'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                'TOA PAYOH', 'WOODLANDS', 'YISHUN']

flat_type_options = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI GENERATION']

flat_model_options = ['2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved',
                      'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette',
                      'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment',
                      'Premium Apartment Loft', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']

mrt_options = sorted(df["Nearest MRT Station"].dropna().unique().tolist())
mall_options = sorted(df["Nearest Mall Name"].dropna().unique().tolist())
npc_options = sorted(df["Nearest NPC"].dropna().unique().tolist())


# --- Streamlit UI ---
st.title("🏠 HDB Resale Price Prediction")

# Row 1: Flat Type + Flat Model
col1, col2 = st.columns(2)
with col1:
    flat_type = st.selectbox("Flat Type", flat_type_options)
with col2:
    flat_model = st.selectbox("Flat Model", flat_model_options)

# Row 2: Floor Area + Storey Level
col3, col4 = st.columns(2)
with col3:
    floor_area = st.number_input("Floor Area (sqm)", 30, 200, 90)
with col4:
    storey_mid = st.number_input("Storey Level", 1, 50, 10)

# Row 3: Remaining Lease + Latest Year of Sale
col5, col6 = st.columns(2)
with col5:
    lease_years = st.number_input("Remaining Lease (Years)", 1, 99, 60)
with col6:
    year = st.number_input("Latest Year of Sale", 2017, 2025, 2021)

# Calculate CPI after `year` is defined
cpi_val = default_df[default_df["year"] == year]["cpi"].mean().round(2)

# Row 4: Lease Commencement + CPI
col7, col8 = st.columns(2)
with col7:
    lease_commence = st.number_input("Lease Commencement Year", 1970, 2023, 2000)
with col8:
    cpi=st.number_input("CPI (Auto-filled by Year)", value=cpi_val, step=0.1, format="%.2f", disabled=True)

# Row 5: Latitude + Longitude
col9, col10 = st.columns(2)
with col9:
    lat = st.number_input("Latitude", 1.2, 1.5, 1.35)
with col10:
    lng = st.number_input("Longitude", 103.6, 104.0, 103.75)

# --- Haversine autofill logic ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

address_data["distance"] = address_data.apply(
    lambda row: haversine(lat, lng, row["Address Lat"], row["Address Long"]), axis=1
)
# Find nearest town from df_merged
default_df["coord_dist"] = default_df.apply(
    lambda row: haversine(lat, lng, row["Address Lat"], row["Address Long"]), axis=1
)

nearest_flat = default_df.loc[default_df["coord_dist"].idxmin()]

nearest = address_data.loc[address_data["distance"].idxmin()]
town = nearest_flat["town"]  # autofill town
mrt_name = nearest["Nearest MRT Station"]
mall_name = nearest["Nearest Mall Name"]
npc_name = nearest["Nearest NPC"]
mrt_distance = nearest["Nearest MRT Distance"]
mall_distance = nearest["Nearest Mall Distance"]
npc_distance = nearest["Nearest NPC Distance"]


with st.expander("📍 Auto-filled Location-Based Features"):
    st.markdown(f"**Detected Town:** {town}")
    st.markdown(f"**Nearest MRT Station:** {mrt_name} ({mrt_distance:.2f} km)")
    st.markdown(f"**Nearest Mall:** {mall_name} ({mall_distance:.2f} km)")
    st.markdown(f"**Nearest NPC:** {npc_name} ({npc_distance:.2f} km)")
    if st.checkbox("View Location Map"):
      st.map(pd.DataFrame({'lat': [lat], 'lon': [lng]}), use_container_width=False, height=250)





# Combine into DataFrame
input_df = pd.DataFrame([{
    "floor_area_sqm": floor_area,
    "remaining_lease_years": lease_years,
    "Nearest MRT Distance": mrt_distance,
    "Nearest Mall Distance": mall_distance,
    "Nearest NPC Distance": npc_distance,
    "cpi": cpi,
    "storey_mid": storey_mid,
    "year": year,
    "lease_commence_date": lease_commence,
    "Address Lat": lat,
    "Address Long": lng,
    "town": town,
    "flat_type": flat_type,
    "flat_model": flat_model,
    "Nearest MRT Station": mrt_name,
    "Nearest Mall Name": mall_name,
    "Nearest NPC": npc_name
}])

# Match preprocessing used in training
input_encoded = pd.get_dummies(input_df)

# Align columns with model's expected structure
expected_cols = pd.read_csv("df_merged.csv").drop(columns=["month", "Address", "resale_price", "resale_price_real"]).copy()
expected_encoded = pd.get_dummies(expected_cols)
final_columns = expected_encoded.columns

for col in final_columns:
    if col not in input_encoded:
        input_encoded[col] = 0
input_encoded = input_encoded[final_columns]

# Scale
input_encoded = input_encoded[scaler.feature_names_in_]  # Match order & drop extras
scaled_input = scaler.transform(input_encoded)
tensor_input = torch.FloatTensor(scaled_input)

# Predict
if st.button("Predict"):
    with torch.no_grad():
        prediction = model(tensor_input).item()
    st.success(f"💰 Predicted Resale Price: ${prediction:,.2f}")
    # Save Prediction
    input_df["predicted_price"] = prediction
    try:
        past = pd.read_csv("prediction_log.csv")
        updated = pd.concat([past, input_df], ignore_index=True)
    except FileNotFoundError:
        updated = input_df
    updated.to_csv("prediction_log.csv", index=False)
    st.success("✅ Prediction saved to log.")
    # Show download button if prediction log exists
    if os.path.exists("prediction_log.csv"):
      with open("prediction_log.csv", "rb") as f:
        st.download_button(
            label="⬇️ Download Prediction Log (CSV)",
            data=f,
            file_name="prediction_log.csv",
            mime="text/csv"
        )


    # Confidence Interval Chart
    lower = prediction * 0.9
    upper = prediction * 1.1
    st.markdown(f" **Estimated Price Range:**")
    st.markdown(f"- **Lower Bound:** ${lower:,.2f}")
    st.markdown(f"- **Upper Bound:** ${upper:,.2f}")

