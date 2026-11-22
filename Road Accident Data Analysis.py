
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("road_accidents.csv")

df = df.drop_duplicates()

df["Date"] = pd.to_datetime(df["Date"])
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour

df["Month"] = df["Date"].dt.month_name()
df["Day"] = df["Date"].dt.day_name()

df["Time_Period"] = np.select(
    [
        df["Time"] < 6,
        df["Time"] < 12,
        df["Time"] < 18
    ],
    ["Night", "Morning", "Afternoon"],
    default="Evening"
)

print("Dataset Overview")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nTotal Accidents:", df["Accident_ID"].nunique())
print("Total Casualties:", df["Casualties"].sum())
print("Average Casualties:", round(df["Casualties"].mean(), 2))

severity_analysis = df["Accident_Severity"].value_counts()
print("\nAccident Severity")
print(severity_analysis)

city_analysis = df["City"].value_counts()
print("\nCity-wise Accidents")
print(city_analysis)

weather_analysis = df["Weather"].value_counts()
print("\nWeather-wise Accidents")
print(weather_analysis)

road_analysis = df["Road_Type"].value_counts()
print("\nRoad Type Analysis")
print(road_analysis)

vehicle_analysis = df["Vehicle_Type"].value_counts()
print("\nVehicle-wise Accidents")
print(vehicle_analysis)

time_analysis = df["Time_Period"].value_counts()
print("\nTime Period Analysis")
print(time_analysis)

monthly_analysis = df.groupby("Month")["Accident_ID"].count()
print("\nMonthly Accidents")
print(monthly_analysis)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Accident_Severity")
plt.title("Accident Severity Distribution")
plt.xlabel("Accident Severity")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="City")
plt.title("City-wise Road Accidents")
plt.xlabel("City")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Weather")
plt.title("Weather-wise Accident Distribution")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Vehicle_Type")
plt.title("Vehicle-wise Accident Analysis")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Time_Period",
              order=["Morning", "Afternoon", "Evening", "Night"])
plt.title("Accidents by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Month",
              order=df["Month"].unique())
plt.title("Monthly Accident Analysis")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="Road_Type", y="Casualties")
plt.title("Average Casualties by Road Type")
plt.xlabel("Road Type")
plt.ylabel("Average Casualties")
plt.tight_layout()
plt.show()

print("\nBusiness Insights")

print("Most Affected City:", city_analysis.idxmax())
print("Most Common Accident Severity:", severity_analysis.idxmax())
print("Most Common Weather Condition:", weather_analysis.idxmax())
print("Most Affected Vehicle Type:", vehicle_analysis.idxmax())
print("Highest Accident Time Period:", time_analysis.idxmax())
print("Month with Most Accidents:", monthly_analysis.idxmax())
