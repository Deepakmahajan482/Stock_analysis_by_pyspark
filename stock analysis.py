# Databricks notebook source
import pyspark
from pyspark.sql import SparkSession

# COMMAND ----------

spark=SparkSession.builder\
    .appName('stock price analysis')\
    .getOrCreate()

# COMMAND ----------

df = spark.read.option("header", True).option("inferSchema", True).csv("dbfs:/Volumes/college_project/stock_analysis/stock/")
df.display()


# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.select('ticker').show(3)

# COMMAND ----------

df.select('ticker','date','open').show(5)

# COMMAND ----------

# |-- Ticker: string (nullable = true)
#  |-- Date: date (nullable = true)
#  |-- Close/Last: string (nullable = true)
#  |-- Volume: integer (nullable = true)
#  |-- Open: string (nullable = true)
#  |-- High: string (nullable = true)
#  |-- Low: string (nullable = true)
df.filter(df.Ticker=="MSFT").show(10)

# COMMAND ----------

from pyspark.sql.functions import col

df.filter((col("Ticker") == "MSFT") & (col("Date") == "2023-05-31")).show()


# COMMAND ----------

df.filter(((df.Ticker=="MSFT") | (df.Ticker=="V")) & (df.Date=="2023-05-31")).show(15)

# COMMAND ----------

df.filter(df.Ticker.isin("MSFT","V") & (col('Date')=="2023-05-31")).show(15)


# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Remove Dollar Sign from close,open and all
# MAGIC

# COMMAND ----------

from pyspark.sql.types import *;
def dollor_remove(x):
    if isinstance(x,str):
        return float(x.strip("$"))
    elif isinstance(x,int) or isinstance(x,float):
        return x
    else:
        return None
    
print(dollor_remove("$10000.91"))
print(dollor_remove(1.21))
    

# COMMAND ----------

parse_number=udf(dollor_remove,FloatType())
df=df.withColumn("open",parse_number(col("open")))\
    .withColumn("Close/Last",parse_number(col("Close/Last")))\
    .withColumn("High",parse_number(col("High")))\
    .withColumn("Low",parse_number(col("Low")))
df.printSchema()


# COMMAND ----------

df.show()

# COMMAND ----------

df=df.withColumnRenamed('Close/Last','Close')

# COMMAND ----------

df.show()

# COMMAND ----------

cleaned_stocks=df.select("Ticker","Date","Volume","Open","Low","High","Close")

# COMMAND ----------

cleaned_stocks.show()

# COMMAND ----------

cleaned_stocks.describe(["Volume","Open","Low","High","Close"]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Calculate the Maximum price and Total Volume for various stocks

# COMMAND ----------

from pyspark.sql.functions import *
cleaned_stocks.groupBy('Ticker').agg(max('open').alias('Maximum'),sum('Volume').alias('TotalVolume')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Calculate maximum price of stocks each year  

# COMMAND ----------

cleaned_stocks=cleaned_stocks.withColumn('Year',year(col('Date')))\
    .withColumn('Month',month(col('Date')))\
        .withColumn('Day',day(col('Date')))\
        .withColumn('Week',weekofyear(col('Date')))
cleaned_stocks.show(20)

yearly=cleaned_stocks.groupBy('Ticker','year').agg(max('Open').alias('Maximum Price'),min('Open').alias('Minimum Price')).orderBy('Ticker')
yearly.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Calculate Maximum price and Minimum price of stocks monthly

# COMMAND ----------

monthly=cleaned_stocks.groupBy('Ticker','Month').agg(max('Open').alias('Maximum Price'),min('Open').alias('Minimum Price')).orderBy('Ticker','month')
monthly.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate the Maximum price and Minium price of Stocks Weekly

# COMMAND ----------

week=cleaned_stocks.groupBy('Ticker','Week').agg(max('Open').alias('Maximum Price'),min('Open').alias('Minimum Price')).orderBy('Ticker','Week')
week.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### JOINS

# COMMAND ----------

historic_stocks=cleaned_stocks.join(yearly,((cleaned_stocks["Ticker"]==yearly["Ticker"])&(cleaned_stocks["Year"]==yearly["year"])),'inner')
historic_stocks.display()

# COMMAND ----------

historic_stocks=cleaned_stocks.join(yearly,((cleaned_stocks["Ticker"]==yearly["Ticker"])&(cleaned_stocks["Year"]==yearly["year"])),'inner').drop(yearly.year,yearly.Ticker)
historic_stocks.display()

# COMMAND ----------

historic_stocks_monthly=cleaned_stocks.join(monthly,((cleaned_stocks["Ticker"]==monthly["Ticker"])&(cleaned_stocks["Month"]==monthly["Month"])),'inner').drop(monthly.Month,monthly.Ticker)
historic_stocks.display()

# COMMAND ----------

historic_stocks.columns

# COMMAND ----------

final_stocks=historic_stocks.select('Ticker','Date','Volume','Open','Low','High','Close','Year','Month','Day','Week','Maximum Price','Minimum Price')
final_stocks.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###SQL Query

# COMMAND ----------

final_stocks.createOrReplaceTempView('Stocks_data')

# COMMAND ----------

spark.sql("select  * from Stocks_data where Ticker='MSFT' and Year=2023 limit 10").display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Advanced Analysis
# MAGIC

# COMMAND ----------

## Calculating moving average
from pyspark.sql.window import Window

df.withColumn('Moving average',avg('Open').over(Window.partitionBy('Ticker').orderBy(col('Date')).rowsBetween(Window.unboundedPreceding, Window.currentRow))).display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving Data 

# COMMAND ----------

## Parquet format
result=final_stocks.select('Ticker','Date','Close','High','Low','Open','Volume','Year','Month','Day','Week','Maximum Price','Minimum Price')

# COMMAND ----------


result.coalesce(1).write \
    .partitionBy("Ticker","Date")\
    .option("header", True) \
    .mode("overwrite") \
    .csv("dbfs:/Volumes/college_project/stock_analysis/stock/processed_csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plot of ticker and company according to year

# COMMAND ----------


from pyspark.sql.functions import year
import matplotlib.pyplot as plt

# Step 1: Add 'Year' column
df = df.withColumn("Year", year("date"))

# 🔹 Step 2: Filter for a specific year (change 2018 to any year you want)
year_to_plot = 2018
df_filtered = df.filter(df["Year"] == year_to_plot)

# Step 3: Select required columns
df_selected = df_filtered.select("ticker", "high")

# Step 4: Convert to Pandas
pdf = df_selected.toPandas()

# Step 5: Group by Ticker and take average High
grouped = pdf.groupby("ticker")["high"].mean().reset_index()

# Step 6: Plot single line chart
plt.figure(figsize=(12, 6))
plt.plot(grouped["ticker"], grouped["high"], marker='o', color='green', label=f'Year {year_to_plot}')

plt.title(f"Average High Price per Ticker - {year_to_plot}")
plt.xlabel("Ticker")
plt.ylabel("Average High Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

