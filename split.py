import os
import argparse

from transformers import AutoTokenizer

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--partitions", type=int, default=500, help="")

  args, unknown = parser.parse_known_args()
  return args, unknown



def main():
  args, unknown = read_args()
  
  conf = pyspark.SparkConf()
  sc = pyspark.SparkContext(conf=conf)
  spark = pyspark.sql.SparkSession(sc)

  LEDtokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
  PXtokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-base")

  def count_LEDtokens(text):
    return LEDtokenizer.encode(text, return_tensors='pt').size()[1]
  def count_PXtokens(text):
    return PXtokenizer.encode(text, return_tensors='pt').size()[1]
  
  count_LEDtokens_udf = F.udf(count_LEDtokens, spark_types.IntegerType())
  count_PXtokens_udf = F.udf(count_PXtokens, spark_types.IntegerType())
  orig_test = os.path.join(args.data_root, 'test.txt')
  orig_val = os.path.join(args.data_root, 'val.txt')
  test_df = spark.read.json(orig_test)
  df = test_df.union(spark.read.json(orig_val)).repartition(args.partitions, "article_id")

  df = df.withColumn("LEDtokens", count_LEDtokens_udf(F.concat_ws(" ", F.col("article_text")))).withColumn("PXtokens", count_PXtokens_udf(F.concat_ws(" ", F.col("article_text"))))
  df = df.where(F.col('LEDtokens') <= 16384)
  df = df.where(F.col('PXtokens') <= 16384)
  df = df.orderBy(F.col('LEDtokens'), F.col('PXtokens'), ascending=False).limit(5000)

  output_log = os.path.join(args.data_root, "logging/log.txt")
  with open(output_log, "a+") as writer:
    wrier.write("------Unsplitted data------\n")
    writer.write("Total entries:", df.count())
    writer.write("avg LED tokens:",  df.select(F.avg(df['LEDtokens'])).collect()[0][0], "\n")
    writer.write("median LED tokens:", df.approxQuantile("LEDtokens", [0.5], 0), "\n")
    writer.write("min LED tokens:",  df.select(F.min(df['LEDtokens'])).collect()[0][0], "\n")
    writer.write("max LED tokens:",  df.select(F.max(df['LEDtokens'])).collect()[0][0], "\n")
    writer.write("avg Pegasus-X tokens:",  df.select(F.avg(df['PXtokens'])).collect()[0][0], "\n")
    writer.write("median LED tokens:", df.approxQuantile("LEDtokens", [0.5], 0), "\n")
    writer.write("median Pegasus-X tokens:", df.approxQuantile("PXtokens", [0.5], 0), "\n")
    writer.write("min Pegasus-X tokens:",  df.select(F.min(df['PXtokens'])).collect()[0][0], "\n")
    writer.write("max Pegasus-X tokens:",  df.select(F.max(df['PXtokens'])).collect()[0][0], "\n")
  
  df = df.orderBy(F.rand())
  rows = df.count()

  train_df = df.limit(round(rows*0.8))
  valid_test_df = df.filter(~df["article_id"].isin(list(train_df.select(train_df.article_id).toPandas()['article_id'])))
  valid_df = valid_test_df.limit(round(valid_test_df.count()*0.5))
  test_df = valid_test_df.filter(~valid_test_df["article_id"].isin(list(valid_df.select(train_df.article_id).toPandas()['article_id'])))

  train_df.write.json(path=os.path.join(args.data_root, "train"), mode="overwrite")
  test_df.write.json(path=os.path.join(args.data_root, "test"), mode="overwrite")
  valid_df.write.json(path=os.path.join(args.data_root, "val"), mode="overwrite")

  os.system('cat ' + args.data_root + '/train/part-* >' + args.data_root + '/train.txt')
  os.system('cat ' + args.data_root + '/val/part-* >' + args.data_root + '/val.txt')
  os.system('cat ' + args.data_root + '/test/part-* >' + args.data_root + '/test.txt')

  os.system('rm -r ' + args.data_root + '/train')
  os.system('rm -r ' + args.data_root + '/val')
  os.system('rm -r ' + args.data_root + '/test')

if __name__ == "__main__":
  main()
