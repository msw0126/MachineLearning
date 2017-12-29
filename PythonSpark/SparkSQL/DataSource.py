#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: DataSource.py 
@desc: SparkSQl的各种数据源文件读取
@time: 2017/12/29 
"""

from __future__ import print_function

from pyspark.sql import SparkSession
from pyspark.sql import Row


def basic_datasource_example(spark):
    df = spark.read.load('F:\project\MachineLearning\PythonSpark\Data\\users.parquet')
    df.select("name", "favorate_color").write.save("namesAndFavCOlors.parquet")



if __name__ == '__main__':
    spark = SparkSession.builder\
                        .appName("Python Spark SQL datasource Example")\
                        .getOrCreate
    basic_datasource_example(spark)
    spark.stop()

