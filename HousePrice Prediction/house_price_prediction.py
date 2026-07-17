import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load & prepare training data ──────────────────────────────────────────
FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
TARGET   = "SalePrice"

train = pd.read_csv("train.csv")

X_train = train[FEATURES].copy()
y_train = train[TARGET].copy()

for col in FEATURES:
    X_train[col] = X_train[col].fillna(X_train[col].median())

# ── 2. Scale & train ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── 3. Show training metrics ──────────────────────────────────────────────────
y_pred_train = model.predict(X_train_scaled)
mae  = mean_absolute_error(y_train, y_pred_train)
rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2   = r2_score(y_train, y_pred_train)

print("=" * 50)
print("   House Price Prediction — Linear Regression")
print("=" * 50)
print(f"  Model trained on {len(train)} houses")
print(f"  R²   : {r2:.4f}")
print(f"  MAE  : ${mae:,.0f}")
print(f"  RMSE : ${rmse:,.0f}")
print("=" * 50)


# ── 4. Helper: safely read a numeric input ────────────────────────────────────
def get_input(prompt, min_val=0, max_val=None, is_int=True):
    while True:
        try:
            raw = input(prompt).strip()
            value = int(raw) if is_int else float(raw)
            if value < min_val:
                print(f"  ⚠  Value must be at least {min_val}. Try again.")
                continue
            if max_val is not None and value > max_val:
                print(f"  ⚠  Value must be at most {max_val}. Try again.")
                continue
            return value
        except ValueError:
            print("  ⚠  Please enter a valid number.")


# ── 5. Interactive prediction loop ────────────────────────────────────────────
print("\nEnter house details below to get a price prediction.")
print("Type 'quit' or 'q' at any prompt to exit.\n")

while True:
    print("-" * 50)
    raw = input("Square footage (living area, e.g. 1500): ").strip().lower()
    if raw in ("q", "quit"):
        print("Goodbye!")
        break
    try:
        sqft = float(raw)
        if sqft <= 0:
            print("  ⚠  Square footage must be positive. Try again.")
            continue
    except ValueError:
        print("  ⚠  Please enter a valid number.")
        continue

    bedrooms  = get_input("Number of bedrooms           : ", min_val=0, max_val=20)
    full_bath = get_input("Number of full bathrooms     : ", min_val=0, max_val=10)
    half_bath = get_input("Number of half bathrooms     : ", min_val=0, max_val=10)

    house = pd.DataFrame([[sqft, bedrooms, full_bath, half_bath]], columns=FEATURES)
    house_scaled = scaler.transform(house)
    predicted_price = model.predict(house_scaled)[0]
    predicted_price = max(predicted_price, 0)  

    print(f"\n  ✔  Predicted Sale Price : ${predicted_price:,.0f}")

    again = input("\nPredict another house? (yes/no): ").strip().lower()
    if again not in ("yes", "y"):
        print("Goodbye!")
        break
