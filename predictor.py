
import pandas as pd
import numpy as np

def classify_investment(nearby_schools, nearby_hospitals, transport, amenity_score, age):
    score = (
        float(nearby_schools) * 0.05 +
        float(nearby_hospitals) * 0.04 +
        float(transport) * 0.06 +
        float(amenity_score) * 0.03 -
        float(age) * 0.002
    )
    THRESHOLD = 0.85
    label = "✅ Good Investment" if score > THRESHOLD else "❌ Not Recommended"
    return label, round(score, 3)


def predict_price_5yr(current_price_lakhs, state, property_type):
    state_growth = {
        "Maharashtra": 0.09, "Karnataka": 0.08, "Tamil Nadu": 0.08,
        "Delhi": 0.10, "Telangana": 0.09, "Gujarat": 0.07,
        "Rajasthan": 0.07, "Uttar Pradesh": 0.07, "West Bengal": 0.07,
        "Punjab": 0.06, "Haryana": 0.08, "Kerala": 0.07,
        "Andhra Pradesh": 0.08, "Madhya Pradesh": 0.06,
    }
    base_rate = state_growth.get(state, 0.07)
    type_multiplier = {
        "Villa": 1.02, "Apartment": 1.0,
        "House": 0.98, "Plot": 1.05
    }.get(property_type, 1.0)
    rate = base_rate * type_multiplier
    price_5yr = float(current_price_lakhs) * ((1 + rate) ** 5)
    return round(price_5yr, 2), round(rate * 100, 2)


def get_investment_summary(score, label, price_now, price_5yr, growth_rate):
    profit = round(price_5yr - price_now, 2)
    roi = round((profit / price_now) * 100, 2)
    summary = {
        "label": label,
        "score": score,
        "price_now": price_now,
        "price_5yr": price_5yr,
        "growth_rate": growth_rate,
        "expected_profit": profit,
        "roi_percent": roi
    }
    return summary
