
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
df = pd.read_csv("data/FUTURE_DS_01_sales_dataset.csv")

# Step 2: Display first 5 records
print("FIRST 5 SALES RECORDS")
print(df.head())

# Step 3: Check dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Step 4: Display column names
print("\nCOLUMN NAMES")
print(df.columns)

# Step 5: Display dataset information
print("\nDATASET INFORMATION")
df.info()

# Step 6: Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 7: Check duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Step 8: Count total rows
print("\nTOTAL ROWS")
print(len(df))


# STEP 7: Convert Order_Date into datetime format

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Check the updated data type
print("\nUPDATED DATE DATA TYPE")
print(df["Order_Date"].dtype)

# Display first 5 dates
print("\nCONVERTED ORDER DATES")
print(df["Order_Date"].head())


# STEP 8: BUSINESS KPI CALCULATIONS

# 1. Total Revenue
total_revenue = df["Sales"].sum()

# 2. Total Profit
total_profit = df["Profit"].sum()

# 3. Total Orders
total_orders = df["Order_ID"].nunique()

# 4. Total Quantity Sold
total_quantity = df["Quantity"].sum()

# 5. Average Order Value
average_order_value = total_revenue / total_orders

# Display business KPIs
print("\n========== BUSINESS PERFORMANCE KPIs ==========")

print("Total Revenue: Rs.", round(total_revenue, 2))
print("Total Profit: Rs.", round(total_profit, 2))
print("Total Orders:", total_orders)
print("Total Quantity Sold:", total_quantity)
print("Average Order Value: Rs.", round(average_order_value, 2))

print("================================================")


# STEP 9: REGIONAL SALES PERFORMANCE

region_analysis = df.groupby("Region").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order_ID", "nunique"),
    Total_Quantity=("Quantity", "sum")
)

# Sort regions by sales
region_analysis = region_analysis.sort_values(
    by="Total_Sales",
    ascending=False
)

print("\n========== REGIONAL SALES PERFORMANCE ==========")
print(region_analysis.round(2))
print("=================================================")


# STEP 10: CATEGORY PERFORMANCE ANALYSIS

category_analysis = df.groupby("Category").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order_ID", "nunique"),
    Total_Quantity=("Quantity", "sum")
)

# Sort categories by revenue
category_analysis = category_analysis.sort_values(
    by="Total_Sales",
    ascending=False
)

print("\n========== CATEGORY PERFORMANCE ==========")
print(category_analysis.round(2))
print("===========================================")


# STEP 11: PRODUCT PERFORMANCE ANALYSIS

product_analysis = df.groupby("Product").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order_ID", "nunique"),
    Total_Quantity=("Quantity", "sum")
)

# Sort products by revenue
product_analysis = product_analysis.sort_values(
    by="Total_Sales",
    ascending=False
)

print("\n========== PRODUCT PERFORMANCE ==========")
print(product_analysis.round(2))
print("=========================================")

# Top 5 products by revenue
print("\nTOP 5 PRODUCTS BY REVENUE")
print(product_analysis.head(5))


# STEP 12: TOP 10 PRODUCTS REVENUE CHART

top_10_products = product_analysis.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_10_products.index,
    top_10_products["Total_Sales"]
)

plt.title("Top 10 Products by Revenue")
plt.xlabel("Total Revenue (Rs.)")
plt.ylabel("Product")

# Display highest revenue product at the top
plt.gca().invert_yaxis()

plt.tight_layout()

# Save chart
plt.savefig("reports/top_10_products.png")

plt.show()


# STEP 13: MONTHLY SALES ANALYSIS

# Extract month from Order_Date
df["Month"] = df["Order_Date"].dt.month

# Group sales by month
monthly_sales = df.groupby("Month").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order_ID", "nunique")
)

print("\n========== MONTHLY SALES PERFORMANCE ==========")
print(monthly_sales.round(2))
print("================================================")

# Identify the best sales month
best_month = monthly_sales["Total_Sales"].idxmax()

print("\nBest Sales Month:", best_month)
print("Highest Monthly Revenue:", monthly_sales["Total_Sales"].max())



# STEP 14: MONTHLY REVENUE TREND CHART

# Use the existing month names
month_names = monthly_sales.index

plt.figure(figsize=(10, 5))

plt.plot(
    month_names,
    monthly_sales["Total_Sales"],
    marker="o"
)

plt.title("Monthly Revenue Trend - 2025")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs.)")

plt.grid(True)
plt.tight_layout()

plt.savefig("reports/monthly_revenue_trend.png")

plt.show()


# STEP 15: SALES CHANNEL PERFORMANCE

channel_analysis = df.groupby("Sales_Channel").agg(
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum"),
    Total_Orders=("Order_ID", "nunique"),
    Total_Quantity=("Quantity", "sum")
)

channel_analysis = channel_analysis.sort_values(
    by="Total_Sales",
    ascending=False
)

print("\n========== SALES CHANNEL PERFORMANCE ==========")
print(channel_analysis.round(2))
print("================================================")


# STEP 16: DISCOUNT ANALYSIS

discount_analysis = df.groupby("Category").agg(
    Average_Discount=("Discount_Percent", "mean"),
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Profit", "sum")
)

discount_analysis = discount_analysis.sort_values(
    by="Average_Discount",
    ascending=False
)

print("\n========== DISCOUNT ANALYSIS ==========")
print(discount_analysis.round(2))
print("=======================================")


# STEP 17: DATA VALIDATION

df["Calculated_Sales"] = (
    df["Quantity"]
    * df["Unit_Price"]
    * (1 - df["Discount_Percent"] / 100)
)

df["Sales_Difference"] = (
    df["Sales"] - df["Calculated_Sales"]
).abs()

incorrect_sales = df[df["Sales_Difference"] > 0.01]

print("\n========== DATA VALIDATION ==========")
print("Total Records:", len(df))
print("Incorrect Sales Records:", len(incorrect_sales))
print("Correct Sales Records:", len(df) - len(incorrect_sales))
print("=====================================")


# STEP 18: REGIONAL SALES CHART

plt.figure(figsize=(9, 5))

plt.bar(
    region_analysis.index,
    region_analysis["Total_Sales"]
)

plt.title("Total Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Revenue (Rs.)")

plt.tight_layout()

plt.savefig("reports/regional_sales.png")

plt.show()


# STEP 19: CATEGORY SALES CHART

plt.figure(figsize=(9, 5))

plt.bar(
    category_analysis.index,
    category_analysis["Total_Sales"]
)

plt.title("Total Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Revenue (Rs.)")

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("reports/category_sales.png")

plt.show()


# STEP 20: SALES CHANNEL COMPARISON CHART

plt.figure(figsize=(8, 5))

plt.bar(
    channel_analysis.index,
    channel_analysis["Total_Sales"]
)

plt.title("Revenue by Sales Channel")
plt.xlabel("Sales Channel")
plt.ylabel("Revenue (Rs.)")

plt.tight_layout()

plt.savefig("reports/sales_channel_comparison.png")

plt.show()




