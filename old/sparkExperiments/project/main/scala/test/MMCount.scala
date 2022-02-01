

package main.scala.test


import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object MMCount {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("MMCount")
      .getOrCreate()


    if (args.length < 1 )  {
      print("usage program filename")
      sys.exit(1)
    }


    val mmFile = args(0)

    val mmDF = spark.read.format('csv')
      .option("header", "true")
      .option("inferSchema", "true")
      .load(mmFile)

      val countMmDF = mmDF
        .select('state', 'color', 'count')
        .groupBy('state', 'color')
        .agg(count('count').alias('total'))
        .orderBy(desc("total"))

      countMmDF.show(60)
      println(s"Total rows = ${countMmDF.count()}")
  }
}
