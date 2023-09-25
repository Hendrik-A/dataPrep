import os
import argparse
import re

import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as spark_types

def read_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_root", type=str, help="")
  parser.add_argument("--partitions", type=int, default=500, help="")

  args, unknown = parser.parse_known_args()
  return args, unknown

def clean_section_names(sections):
  cleaned = [None] * len(sections)
  for i in range(len(sections)):
    cleaned[i] = re.sub("\[sec\d*\]", "", sections[i])
  return cleaned


def main():
  args, unknown = read_args()
  
  conf = pyspark.SparkConf()
  sc = pyspark.SparkContext(conf=conf)
  spark = pyspark.sql.SparkSession(sc)

  log_dir = os.path.join(args.data_root, "logging")
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  log_file = os.path.join(log_dir, "log.txt")

  clean_section_names_udf = F.udf(clean_section_names, spark_types.ArrayType(spark_types.StringType()))
  
  data_path = os.path.join(args.data_root, 'countedTokens.txt')
  df = spark.read.json(data_path).repartition(500, "article_id")

  df = df.where(F.col('LEDtokens') <= 16384)
  df = df.where(F.col('PXtokens') <= 16384)
  df = df.orderBy(F.col('LEDtokens'), F.col('PXtokens'), ascending=False).limit(5000)
  df = df.withColumn("section_names", clean_section_names_udf("section_names"))
  
  with open(log_file, "a+") as writer:
    writer.write("------Unsplitted data statistics------\n")
    writer.write("Total entries:"+str(df.count())+"\n")
    writer.write("avg LED tokens:"+str(df.select(F.avg(df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("min LED tokens:"+str(df.select(F.min(df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("max LED tokens:"+str(df.select(F.max(df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("avg Pegasus-X tokens:"+str(df.select(F.avg(df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("median Pegasus-X tokens:"+str(df.approxQuantile("PXtokens", [0.5], 0))+"\n")
    writer.write("min Pegasus-X tokens:"+str(df.select(F.min(df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("max Pegasus-X tokens:"+str(df.select(F.max(df['PXtokens'])).collect()[0][0])+"\n")
  
  df = df.orderBy(F.rand())
  rows = df.count()

  train_df = df.limit(round(rows*0.8))
  valid_test_df = df.filter(~df["article_id"].isin(list(train_df.select(train_df.article_id).toPandas()['article_id'])))
  valid_df = valid_test_df.limit(round(valid_test_df.count()*0.5))
  test_df = valid_test_df.filter(~valid_test_df["article_id"].isin(list(valid_df.select(train_df.article_id).toPandas()['article_id'])))

  train_df.write.json(path=os.path.join(args.data_root, "train"), mode="overwrite")
  test_df.write.json(path=os.path.join(args.data_root, "test"), mode="overwrite")
  valid_df.write.json(path=os.path.join(args.data_root, "val"), mode="overwrite")

  with open(log_file, "a+") as writer:
    writer.write("------Train statistics------\n")
    writer.write("Total entries:"+str(str(train_df.count()))+"\n")
    writer.write("avg LED tokens:"+str(train_df.select(F.avg(train_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(train_df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("min LED tokens:"+str(train_df.select(F.min(train_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("max LED tokens:"+str(train_df.select(F.max(train_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("avg Pegasus-X tokens:"+str(train_df.select(F.avg(train_df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(train_df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("median Pegasus-X tokens:"+str(train_df.approxQuantile("PXtokens", [0.5], 0))+"\n")
    writer.write("min Pegasus-X tokens:"+str(train_df.select(F.min(train_df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("max Pegasus-X tokens:"+str(train_df.select(F.max(train_df['PXtokens'])).collect()[0][0])+"\n")

    writer.write("------Validation statistics------\n")
    writer.write("Total entries:" + str(valid_df.count())+"\n")
    writer.write("avg LED tokens:"+str(valid_df.select(F.avg(valid_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(valid_df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("min LED tokens:"+str(valid_df.select(F.min(valid_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("max LED tokens:"+str(valid_df.select(F.max(valid_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("avg Pegasus-X tokens:"+str(valid_df.select(F.avg(valid_df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(valid_df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("median Pegasus-X tokens:"+str(valid_df.approxQuantile("PXtokens", [0.5], 0))+"\n")
    writer.write("min Pegasus-X tokens:"+str(valid_df.select(F.min(valid_df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("max Pegasus-X tokens:"+str(valid_df.select(F.max(valid_df['PXtokens'])).collect()[0][0])+"\n")
    
    writer.write("------Test statistics------\n")
    writer.write("Total entries:"+str(test_df.count()))
    writer.write("avg LED tokens:"+str(test_df.select(F.avg(test_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(test_df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("min LED tokens:"+str(test_df.select(F.min(test_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("max LED tokens:"+str(test_df.select(F.max(test_df['LEDtokens'])).collect()[0][0])+"\n")
    writer.write("avg Pegasus-X tokens:"+str(test_df.select(F.avg(test_df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("median LED tokens:"+str(test_df.approxQuantile("LEDtokens", [0.5], 0))+"\n")
    writer.write("median Pegasus-X tokens:"+str(test_df.approxQuantile("PXtokens", [0.5], 0))+"\n")
    writer.write("min Pegasus-X tokens:"+str(test_df.select(F.min(test_df['PXtokens'])).collect()[0][0])+"\n")
    writer.write("max Pegasus-X tokens:"+str(test_df.select(F.max(test_df['PXtokens'])).collect()[0][0])+"\n")
    
  os.system('cat ' + args.data_root + '/train/part-* >' + args.data_root + '/train.txt')
  os.system('cat ' + args.data_root + '/val/part-* >' + args.data_root + '/val.txt')
  os.system('cat ' + args.data_root + '/test/part-* >' + args.data_root + '/test.txt')

  os.system('rm -r ' + args.data_root + '/train')
  os.system('rm -r ' + args.data_root + '/val')
  os.system('rm -r ' + args.data_root + '/test')

if __name__ == "__main__":
  main()
