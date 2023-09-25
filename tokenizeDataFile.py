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

  output_dir = os.path.join(args.data_root, "counting")
  if not os.path.exists(task_output_dir):
    os.makedirs(task_output_dir)

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
  df.write.json(path=output_dir, mode="overwrite")

  os.system('cat ' + output_dir + '/part-* >' + args.data_root + '/countedTokens.txt')
  os.system('rm -r ' + output_dir)

if __name__ == "__main__":
  main()
