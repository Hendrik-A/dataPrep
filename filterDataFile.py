import os
import argparse
import re

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

KEYWORDS = {
    'introduction': 'i',
    'case': 'i',
    'purpose': 'i',
    'objective': 'i',
    'objectives': 'i',
    'aim': 'i',
    'summary': 'i',
    'findings': 'l',
    'background': 'i',
    'background/aims': 'i',
    'literature': 'l',
    'studies': 'l',
    'related': 'l',
    'methods': 'm',
    'method': 'm',
    'techniques': 'm',
    'methodology': 'm',
    'results': 'r',
    'result': 'r',
    'experiment': 'r',
    'experiments': 'r',
    'experimental': 'r',
    'discussion': 'c',
    'limitations': 'd',
    'conclusion': 'c',
    'conclusions': 'c',
    'concluding': 'c'}

def section_match(keywords):
    def section_match_(sections):
        match = False
        for section in sections:
            section = section.lower().split()
            for wrd in section:
                try:
                    match = KEYWORDS[wrd]
                except KeyError:
                    continue
        return 1 if match else 0
    return F.udf(section_match_, spark_types.ByteType())

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

  b_keywords = sc.broadcast(KEYWORDS)

  df = spark.read.json(args.data_root, "CountedTokens.txt").repartition(args.partitions, "article_id")

  df = df.where(F.col("LEDtextT") <= 16384)
  df = df.where(F.col("PXtextT") <= 16384)
  df = df.withColumn("match", section_match(b_keywords)("section_names")).where(F.col("match") == True).drop("match")
  df = df.orderBy(F.col("LEDtextT"), F.col("PXtextT"), ascending=False).limit(5000).drop("LEDtextT", "LEDabsT", "PXtextT", "PXabsT").orderBy(F.rand())
  
  df.write.json(path=output_dir, mode="overwrite")

  os.system("cat " + output_dir + "/part-* >" + args.data_root + "/countedTokens.txt")
  os.system("rm -r " + output_dir)

if __name__ == "__main__":
  main()
