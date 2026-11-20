import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ==========================================
# 1. EMBEDDED DATASET
# ==========================================

data = pd.DataFrame({
    "Year": [
        2010, 2011, 2012, 2013, 2014,
        2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023, 2024
    ],

    "Cars_On_Road": [
        180000, 195000, 211000, 228000, 246000,
        265000, 285000, 307000, 330000, 354000,
        371000, 398000, 427000, 458000, 491000
    ]
})


# ==========================================
# 2. PREPARE DATA
# ==========================================

X = data[["Year"]]
y = data["Cars_On_Road"]


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 4. TRAIN LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 5. MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE       : {mae:,.2f}")
print(f"RMSE      : {rmse:,.2f}")
print(f"R² Score  : {r2:.4f}")
print(f"Intercept : {model.intercept_:,.2f}")
print(f"Yearly Growth (Slope): {model.coef_[0]:,.2f}")


# ==========================================
# 6. VISUALIZATION
# ==========================================

line_years = pd.DataFrame({
    "Year": range(2010, 2025)
})

line_predictions = model.predict(line_years)

plt.figure(figsize=(10, 6))

plt.scatter(
    data["Year"],
    data["Cars_On_Road"],
    label="Actual Data"
)

plt.plot(
    line_years["Year"],
    line_predictions,
    label="Linear Regression Line"
)

plt.xlabel("Year")
plt.ylabel("Cars on Road")
plt.title("Cars on Road Prediction - Linear Regression")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("regression_plot.png", dpi=150)
plt.show()


# ==========================================
# 7. USER INPUT PREDICTION
# ==========================================

try:
    year = int(input("\nEnter a year to predict cars on road: "))

    result = model.predict(
        pd.DataFrame({"Year": [year]})
    )[0]

    print(f"\nEstimated Cars on Road in {year}: {result:,.0f}")

    if year < 2010 or year > 2024:
        print("Note: This prediction is outside the dataset range.")

except ValueError:
    print("Please enter a valid whole-number year.")