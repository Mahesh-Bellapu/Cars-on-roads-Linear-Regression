# Year vs Cars on Road — Linear Regression

A small beginner-friendly machine-learning project that predicts the number of cars on the road from the year using **Simple Linear Regression**.

> **Important:** `data/cars_on_road.csv` contains illustrative sample data for learning and demonstration. It is not official vehicle-registration data. Replace it with a trusted dataset before making real-world claims.

## Project structure

```text
year_vs_cars_linear_regression/
├── data/
│   └── cars_on_road.csv
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. What the project does

- Loads yearly car-count data from a CSV file.
- Splits the data into training and testing sets.
- Trains a Linear Regression model.
- Evaluates it using MAE, RMSE, and R².
- Draws a chart showing actual values and the fitted regression line.
- Predicts the car count for a year entered by the user.

## 2. Run it locally

Install Python 3.9 or newer, then run:

```bash
pip install -r requirements.txt
python train.py
```

Enter a year when prompted. The chart is saved as `regression_plot.png`.

## 3. Dataset format

The CSV must contain these columns:

- `Year` — calendar year
- `Cars_On_Road` — number of cars on the road

You can replace the sample CSV with official data, keeping the same column names.

## 4. Upload to GitHub

1. Create a new repository on GitHub, for example `year-vs-cars-linear-regression`.
2. Extract this ZIP file.
3. Open a terminal in the extracted project folder.
4. Run:

```bash
git init
git add .
git commit -m "Add year vs cars linear regression project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/year-vs-cars-linear-regression.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## 5. Model idea

The model learns a straight-line relationship between year (X) and cars on road (y):

`Predicted cars = intercept + (slope × Year)`

This is a simple demonstration. Real vehicle counts may not follow a straight-line trend, and predictions outside the observed year range can be unreliable.
