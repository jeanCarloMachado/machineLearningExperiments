#!/usr/bin/env python
import sys


import grimoire as g
from pyspark.sql import SparkSession
from pyspark.sql.functions import count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage requires a file name", file=sys.stderr)
        sys.exit(-1)

    spark = (SparkSession.builder.appName("PythonMmCount").getOrCreate())
    mm_file = sys.argv[1]

    mm_df = (spark.read.format('csv').option("header", "true").option("inferSchema", "true")
             .load(mm_file))

    count_mm_df = (mm_df.select("state", "color", "count")
                   .groupBy("state", "color")
                   .agg(count("count").alias("total"))
                   .orderBy("total", ascending=False))

    count_mm_df.show(n = 60, truncate = False)
    print("Total rows=  %d" % (count_mm_df.count()))

    count_mm_one_df = (mm_df.select("state", "color", "count")
        .where(mm_df.state == 'state5')
        .groupBy("State", "Color")
        .agg(count("Count").alias("Total"))
        .orderBy("Total", ascending = False))

    count_mm_one_df.show(n=60, truncate=False)
    print("Total rows=  %d" % (count_mm_one_df.count()))



    # spark.stop()
