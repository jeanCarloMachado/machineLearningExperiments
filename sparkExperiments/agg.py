#!/usr/bin/env python


from pyspark.sql import SparkSession
from pyspark.sql.functions import avg


spark = (SparkSession.builder.appName('AuthorsAge').getOrCreate())


data_df = spark.createDataFrame([('Brooke', 30), ('Denny', 31), ('Jules', 30), ('TD', 35), ('Brooke', 31)], ['name', 'age'])

avg_df = data_df.groupBy('name').agg(avg('age'))


avg_df.show()


