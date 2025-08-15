from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json
from datetime import datetime
import os

app = Flask(__name__, static_folder='static', template_folder='UI_4.0_templates')

MODEL_PATH = "xgb_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place xgb_model.pkl next to app.py")

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict(): 
    try:
        data = request.get_json()
        # Extract and validate inputs
        bed = int(data.get("beds"))
        bath = int(data.get("baths"))
        sqft = int(data.get("sqft"))
        acre_lot = float(data.get("lot"))
        zip_code = str(data.get("zip"))
        # date input is "YYYY-MM" from <input type="month">
        date_str = data.get("date", "2000-01")
        if "-" in date_str:
            year, month = map(int, date_str.split("-"))
        else:
            # fallback
            year, month = 2000, 1

        # Build DataFrame matching the training order & names used earlier
        input_df = pd.DataFrame(
            [[bed, bath, acre_lot, zip_code, sqft, year, month]],
            columns=['bed', 'bath', 'acre_lot', 'zip_code', 'house_size', 'year', 'month']
        )
        # Ensure zip_code is categorical like training
        input_df['zip_code'] = input_df['zip_code'].astype('category')

        # Predict
        price = model.predict(input_df)
        # Convert numpy types to python for JSON
        price_val = float(price[0])

        # Adjusting for inflation manually
        today = datetime.today()
        current_year = today.year
        current_month = today.month
        total_months = (current_year - year) * 12 + (current_month - month)
        inflation_rate = -0.002
        price_val *= (1 + inflation_rate * total_months)
        
        # Adjusting for Zip Codes manually
        with open("Dataset/zip3_multipliers_corrected.json", "r") as f:
             zip3_multipliers = json.load(f)
        zip_str = str(zip_code).zfill(5)
        zip3 = zip_str[:3]
        multiplier = zip3_multipliers.get(zip3, 1.0)
        multiplier = multiplier*1.7
        price_val*=multiplier

        # Adjusting for Lot Size manually
        price_val = price_val * (1.2) ** acre_lot

        # Return the final price
        return jsonify({"success": True, "price": price_val})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8501)
