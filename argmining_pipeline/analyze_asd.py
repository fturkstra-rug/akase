from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

df = spark.read.option("recursiveFileLookup", "true").parquet("../../main_asd")

stats = df.agg(
    F.count("*").alias("total"),
    F.sum("is_arg").alias("arg_count")
).collect()[0]

total = stats["total"]
arg_count = stats["arg_count"]

print("Total rows:", total)
print("Argument rows:", arg_count)
print("Percentage:", arg_count / total * 100 if total else 0)
