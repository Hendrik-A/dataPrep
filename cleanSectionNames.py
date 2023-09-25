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

    train_data = os.path.join(args.data_root, 'train.txt')
    val_data = os.path.join(args.data_root, 'val.txt')
    test_data = os.path.join(args.data_root, 'test.txt')

    conf = pyspark.SparkConf()
    sc = pyspark.SparkContext(conf=conf)
    spark = pyspark.sql.SparkSession(sc)

    clean_section_names_udf = F.udf(clean_section_names, spark_types.ArrayType(spark_types.StringType()))
    data_prefixes = ['train', 'val', 'test']
    data_paths = [train_data, val_data, test_data]
    task_output_dir = args.data_root

    for data_path, prefix in zip(data_paths, data_prefixes):
        df = spark.read.json(data_path) \
            .repartition(args.partitions, "article_id")

        df = df.withColumn("section_names", clean_section_names_udf("section_names"))

        df.write.json(
            path=os.path.join(task_output_dir, prefix),
            mode="overwrite")

        print(f"Finished writing {prefix} split to {task_output_dir}")

    os.system('cat ' + args.data_root + '/train/part-* >' + args.data_root + '/train.txt')
    os.system('cat ' + args.data_root + '/val/part-* >' + args.data_root + '/val.txt')
    os.system('cat ' + args.data_root + '/test/part-* >' + args.data_root + '/test.txt')

    os.system('rm -r ' + args.data_root + '/train')
    os.system('rm -r ' + args.data_root + '/val')
    os.system('rm -r ' + args.data_root + '/test')


if __name__ == "__main__":
    main()
