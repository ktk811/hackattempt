###########################################
# flask_app.py
# A Flask web API with integrated MongoDB and an AI model.
#  - Provides endpoints for login, registration, and data retrieval.
#  - Retains all helper functions from your original Streamlit code.
###########################################

# --- Dummy modules for Windows compatibility (inject before any other imports) ---
import sys, types, io
dummy_fcntl = types.ModuleType("fcntl")
def ioctl(fd, request, arg=0, mutate_flag=True):
    return 0
dummy_fcntl.ioctl = ioctl
sys.modules["fcntl"] = dummy_fcntl

dummy_termios = types.ModuleType("termios")
dummy_termios.TIOCGWINSZ = 0
sys.modules["termios"] = dummy_termios

sys.modules["StringIO"] = io
# --- End Dummy modules ---

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from urllib.parse import quote_plus
import requests, numpy as np, pandas as pd, datetime
import ee, pymongo, bcrypt
import pickle

# For AI model training
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a strong secret key

# ---------------------------
# EARTH ENGINE INITIALIZATION
# ---------------------------
try:
    ee.Initialize(project='ee-kartik081105')
except Exception as e:
    # Remove interactive authentication for API deployment.
    print("EE initialization failed. Have you run 'earthengine authenticate' externally?", e)
    # Optionally, you can decide to continue with limited functionality.
    
# ---------------------------
# SET UP MONGODB CONNECTION
# ---------------------------
username_db = quote_plus("soveetprusty")
password_db = quote_plus("@Noobdamaster69")
connection_string = f"mongodb+srv://{username_db}:{password_db}@cluster0.bjzstq0.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(connection_string)
db = client["agr_app"]
farmers_col = db["farmers"]
crop_inventory_col = db["crop_inventory"]
pesticide_inventory_col = db["pesticide_inventory"]

# ---------------------------
# USER SETTINGS & INVENTORY DEFAULTS
# ---------------------------
GOOGLE_MAPS_EMBED_API_KEY = "AIzaSyAWHIWaKtmhnRfXL8_FO7KXyuWq79MKCvs"  # Replace with your key
default_crop_prices = {"Wheat": 20, "Rice": 25, "Maize": 18, "Sugarcane": 30, "Cotton": 40}
soil_types = ["Sandy", "Loamy", "Clay", "Silty"]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def get_weather_data(city_name):
    geo_url = "https://nominatim.openstreetmap.org/search"
    params_geo = {"city": city_name, "country": "India", "format": "json"}
    r_geo = requests.get(geo_url, params=params_geo, headers={"User-Agent": "Mozilla/5.0"})
    if r_geo.status_code != 200 or not r_geo.json():
        return None, None, None, None, None, None
    geo_data = r_geo.json()[0]
    lat = float(geo_data["lat"])
    lon = float(geo_data["lon"])
    weather_url = "https://api.open-meteo.com/v1/forecast"
    params_weather = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "hourly": "precipitation",
        "timezone": "Asia/Kolkata"
    }
    r_wth = requests.get(weather_url, params=params_weather)
    if r_wth.status_code != 200:
        return None, None, lat, lon, None, None
    wdata = r_wth.json()
    current_temp = wdata["current_weather"]["temperature"]
    current_time = wdata["current_weather"]["time"]
    hourly_times = wdata["hourly"]["time"]
    hourly_precip = wdata["hourly"]["precipitation"]
    current_precip = hourly_precip[hourly_times.index(current_time)] if current_time in hourly_times else 0
    return current_temp, current_precip, lat, lon, hourly_precip, hourly_times

def get_real_ndvi(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    region = point.buffer(5000)
    today = datetime.date.today()
    start_date = str(today - datetime.timedelta(days=30))
    end_date = str(today)
    s2 = ee.ImageCollection('COPERNICUS/S2') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    s2 = s2.map(add_ndvi)
    ndvi_image = s2.select('NDVI').median()
    ndvi_dict = ndvi_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=region, scale=30)
    ndvi_value = ee.Number(ndvi_dict.get('NDVI')).getInfo()
    return ndvi_value

def get_fertilizer_pesticide_recommendations(ndvi, soil_type):
    if ndvi < 0.5:
        fert = "High NPK mix (Urea, DAP, MOP)"
        pest = "Broad-spectrum insecticide (e.g., Chlorpyrifos)"
    elif ndvi < 0.7:
        fert = "Moderate NPK mix (Balanced fertilizer)"
        pest = "Targeted pesticide (e.g., Imidacloprid)"
    else:
        fert = "Minimal fertilizer needed"
        pest = "No pesticide required"
    if soil_type == "Sandy":
        fert += " (Add extra organic matter & water)"
    elif soil_type == "Clay":
        fert += " (Ensure drainage, avoid overwatering)"
    elif soil_type == "Loamy":
        fert += " (Balanced approach)"
    elif soil_type == "Silty":
        fert += " (Moderate water-holding capacity)"
    return fert, pest

def get_soil_type(lat, lon):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {"lat": lat, "lon": lon, "property": "sand,clay,silt", "depth": "0-5cm"}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
        layers = data.get("properties", {}).get("layers", [])
        sand = clay = silt = None
        for layer in layers:
            name = layer.get("name", "").lower()
            if not layer.get("depths"):
                continue
            mean_val = layer["depths"][0].get("values", {}).get("mean", None)
            if mean_val is None:
                continue
            if "sand" in name:
                sand = mean_val
            elif "clay" in name:
                clay = mean_val
            elif "silt" in name:
                silt = mean_val
        if sand is None or clay is None or silt is None:
            return None
        if sand >= clay and sand >= silt:
            return "Sandy"
        elif clay >= sand and clay >= silt:
            return "Clay"
        elif silt >= sand and silt >= clay:
            return "Silty"
        else:
            return "Loamy"
    except Exception:
        return None

def reverse_geocode(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1}
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code == 200:
        return r.json().get("display_name", "Address not available")
    return "Address not available"

def get_live_shop_list(lat, lon):
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node(around:10000, {lat}, {lon})["shop"];
    out body;
    """
    r = requests.post(overpass_url, data=query)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    elements = data.get("elements", [])
    keywords = ["agro", "farm", "agr", "hort", "garden", "agriculture"]
    exclusions = ["clothes", "apparel", "fashion", "footwear"]
    shops = []
    for elem in elements:
        tags = elem.get("tags", {})
        name = tags.get("name", "").strip()
        shop_tag = tags.get("shop", "").strip()
        if not name:
            continue
        if any(exc in name.lower() for exc in exclusions):
            continue
        if not (any(k in name.lower() for k in keywords) or any(k in shop_tag.lower() for k in keywords)):
            continue
        addr_full = tags.get("addr:full", "").strip()
        if addr_full:
            address = addr_full
        else:
            address_parts = []
            if tags.get("addr:housenumber", "").strip():
                address_parts.append(tags.get("addr:housenumber", "").strip())
            if tags.get("addr:street", "").strip():
                address_parts.append(tags.get("addr:street", "").strip())
            if tags.get("addr:city", "").strip():
                address_parts.append(tags.get("addr:city", "").strip())
            if address_parts:
                address = ", ".join(address_parts)
            else:
                address = reverse_geocode(elem.get("lat"), elem.get("lon"))
        shops.append({"Name": name, "Type": shop_tag, "Address": address})
    df = pd.DataFrame(shops)
    if not df.empty:
        df.index = np.arange(1, len(df) + 1)
        df.index.name = "No."
    return df

def style_shops_dataframe(shops_df):
    shops_df_renamed = shops_df.rename(columns={"Name": "Shop Name", "Type": "Category", "Address": "Full Address"})
    styled_df = shops_df_renamed.style.set_properties({"border": "1px solid #444", "padding": "6px"})\
                           .set_table_styles([
                               {"selector": "th", "props": [("background-color", "#2c2c2c"),
                                                            ("font-weight", "bold"),
                                                            ("text-align", "center"),
                                                            ("color", "#e0e0e0")]},
                               {"selector": "td", "props": [("text-align", "left"),
                                                            ("vertical-align", "top"),
                                                            ("color", "#e0e0e0")]}
                           ])
    return styled_df

# ---------------------------
# AI MODEL TRAINING & PREDICTION
# ---------------------------
def train_ai_model(n_samples=200, random_state=42):
    np.random.seed(random_state)
    ndvi = np.random.uniform(0.3, 0.9, n_samples)
    soil = np.random.randint(0, 4, n_samples)
    X = np.column_stack((ndvi, soil))
    y_fert = np.where(ndvi < 0.5, 0, np.where(ndvi < 0.7, 1, 2))
    y_pest = np.where(ndvi < 0.5, 0, np.where(ndvi < 0.8, 1, 2))
    y = np.column_stack((y_fert, y_pest))
    base_clf = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model = MultiOutputClassifier(base_clf)
    model.fit(X, y)
    return model

ai_model = train_ai_model()

fert_mapping = {
    0: "High NPK mix (Urea, DAP, MOP)",
    1: "Moderate NPK mix (Balanced fertilizer)",
    2: "Minimal fertilizer needed"
}
pest_mapping = {
    0: "Broad-spectrum insecticide (e.g., Chlorpyrifos)",
    1: "Targeted pesticide (e.g., Imidacloprid)",
    2: "No pesticide required"
}

def predict_fertilizer_pesticide(ndvi, soil_type):
    soil_mapping = {"Sandy": 0, "Loamy": 1, "Clay": 2, "Silty": 3}
    soil_value = soil_mapping.get(soil_type, 1)
    feature = np.array([[ndvi, soil_value]])
    preds = ai_model.predict(feature)
    fert_pred = int(preds[0][0])
    pest_pred = int(preds[0][1])
    fertilizer = fert_mapping.get(fert_pred, "Moderate NPK mix (Balanced fertilizer)")
    pesticide = pest_mapping.get(pest_pred, "Targeted pesticide (e.g., Imidacloprid)")
    return fertilizer, pesticide

# ---------------------------
# AUTHENTICATION FUNCTIONS
# ---------------------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_farmer(username, password):
    if farmers_col.find_one({"username": username}):
        return False, "Username already exists."
    hashed_pw = hash_password(password)
    farmers_col.insert_one({"username": username, "password": hashed_pw})
    return True, "Registration successful."

def login_farmer(username, password):
    user = farmers_col.find_one({"username": username})
    if user and check_password(password, user["password"]):
        return True, "Login successful."
    return False, "Invalid username or password."

# ---------------------------
# FLASK ROUTES
# ---------------------------
@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("main"))
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    username_input = request.form["username"]
    password_input = request.form["password"]
    success, msg = login_farmer(username_input, password_input)
    if success:
        session["username"] = username_input
        flash("Login successful!", "success")
        return redirect(url_for("main"))
    else:
        flash(msg, "danger")
        return redirect(url_for("home"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username_input = request.form["username"]
        password_input = request.form["password"]
        success, msg = register_farmer(username_input, password_input)
        if success:
            flash(msg + " Please log in.", "success")
            return redirect(url_for("home"))
        else:
            flash(msg, "danger")
            return render_template("register.html")
    return render_template("register.html")

@app.route("/main")
def main():
    if "username" not in session:
        return redirect(url_for("home"))
    username = session["username"]
    weather = get_weather_data("Mumbai")
    try:
        ndvi = get_real_ndvi(19.0760, 72.8777)  # Coordinates for Mumbai
    except Exception as e:
        ndvi = None
    if ndvi is not None:
        fert, pest = get_fertilizer_pesticide_recommendations(ndvi, "Loamy")
    else:
        fert, pest = "N/A", "N/A"
    context = {
        "username": username,
        "weather": weather,  # (temp, current_rain, lat, lon, hourly_precip, hourly_times)
        "ndvi": ndvi,
        "fertilizer": fert,
        "pesticide": pest,
        "google_maps_key": GOOGLE_MAPS_EMBED_API_KEY,
    }
    return render_template("main.html", **context)

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.config["SESSION_TYPE"] = "filesystem"
    app.run(debug=True)
