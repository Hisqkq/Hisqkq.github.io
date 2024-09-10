---
title: Big Data Analysis with Apache Spark
description: A project to analyze web crawl data with Apache Spark. The analysis focuses on wikipedia images data.
author: <author_id>
date: 2023-06-30 11:33:00 +0800
categories: [University Project, Big Data]
tags: [Spark, WebCrawl, Big Data, Scala, Hadoop]
pin: true
math: true
mermaid: true
image:
  path: /assets/img/BigData/cluster.jpg
  alt: Cluster of computers schematics
---

# Big Data Analysis with Apache Spark

## Introduction

During the **Big Data course** at **Radboud University**, I learned to use `Apache Spark` for large-scale data analysis and `Spark Structured Streaming` for real-time data processing.  
In this presentation, I will introduce **Apache Spark**, **Spark Structured Streaming**, and explain how I used `Spark` to analyze web crawl data, with a focus on image data from **Wikipedia**.

## Big Data & Web Crawling

### What is Big Data?

**Big Data** refers to vast amounts of structured and unstructured data generated daily. The significance lies not just in the volume but in how businesses analyze and derive insights from it to make informed decisions. **Big Data analytics** allows organizations to uncover trends, make predictions, and enhance strategies.

### Web Crawling

**Web crawling** is the automated process of collecting data from websites. It is primarily used by search engines to index web pages, making content searchable. These automated bots, also known as **web crawlers** or **spiders**, systematically browse the internet, collecting information such as **URLs**, **titles**, and **content**.

### Web Crawl Data

**Web crawl data** is a collection of the data harvested by web crawlers. It can include elements such as **URLs**, **metadata**, and page content. This type of data is useful for **search engine optimization**, **web analytics**, and **data mining**.  
In this project, I analyzed web crawl data from **Wikipedia** to extract information related to images. The data, typically stored in a distributed file system like `HDFS` (Hadoop Distributed File System), was stored in the **University's Cluster** using `HDFS` for distributed storage.

## Apache Spark

`Apache Spark` is a distributed computing system designed for large-scale data processing. It offers a unified framework for handling **data parallelism** and **fault tolerance**.  
Spark supports high-level APIs in **Java**, **Scala**, **Python**, and **R**, making it accessible to various programming communities. Its optimized execution engine supports general **execution graphs**, making it efficient for a wide range of tasks. Some of the key components of Spark include:
- **`Spark SQL`**: For processing structured data using SQL-like queries.
- **`MLlib`**: A library for scalable machine learning.
- **`GraphX`**: For graph processing and analysis.
- **`Spark Streaming`**: For real-time data processing.

In this project, I used **Spark** to analyze **Wikipedia's web crawl image data**, focusing on the distributed processing capabilities of Spark and leveraging **`Spark Structured Streaming`** for handling continuous data streams in real time.


### Spark Basics

#### Spark Architecture

Spark is built around the concept of a resilient distributed dataset (RDD). What is RDD ?

<div class="box-info" markdown="1">
<div class="title"> Resilient Distributed Dataset (RDD) </div>
- RDD is a fault-tolerant collection of elements that can be operated on in parallel.
- RDDs can be created from Hadoop InputFormats (such as HDFS files) or by transforming other RDDs.
- RDDs support two types of operations: transformations, which create a new dataset from an existing one, and actions, which return a value to the driver program after running a computation on the dataset.
- RDDs are lazily evaluated, meaning that their values are not computed until they are used in an action.

RDD is the fundamental data structure of Spark. It is an immutable distributed collection of objects. Each dataset in Spark is split into logical partitions, which may be computed on different nodes of the cluster. RDDs can contain any type of Python, Java, or Scala objects, including user-defined classes.

</div>

The following code snippet shows how to create an RDD from a list of numbers and then perform a transformation on it:

```scala
val data = Array(1, 2, 3, 4, 5)
val distData = sc.parallelize(data)
val result = distData.map(x => x * x)
```

In this example, `sc.parallelize(data)` creates an RDD from the data array. The `map` transformation is then applied to the RDD to square each element.

#### Spark Session

The entry point to programming Spark with the Dataset and DataFrame API. A SparkSession can be used to create DataFrame, register DataFrame as tables, execute SQL over tables, cache tables, and read parquet files.

```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder()
  .appName("Spark SQL basic example")
  .config("spark.some.config.option", "some-value")
  .getOrCreate()
```

Then you can use the `spark` object to create DataFrames and perform operations on them. For example:

```scala
val df = spark.read.json("examples/src/main/resources/people.json")
df.show()
```

You can also run SQL queries over tables that you register with the `spark` object. For example:

```scala
df.createOrReplaceTempView("people")
val sqlDF = spark.sql("SELECT * FROM people")
sqlDF.show()
```

### Spark Structured Streaming

Spark Structured Streaming is a scalable and fault-tolerant stream processing engine built on the Spark SQL engine. You can express your streaming computation the same way you would express a batch computation on static data. The Spark SQL engine will take care of running it incrementally and continuously and updating the final result as streaming data continues to arrive.

```scala
package org.rubigdata

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.scalalang.typed


object RUBigDataApp {
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("RUBigDataApp").getOrCreate()
    import spark.implicits._ 
    spark.sparkContext.setLogLevel("WARN")
    val regex = "^([A-Z].+) ([A-Z].+) was sold for (\\d+)gp$"
    val socketDF = spark.readStream
      .format("socket")
      .option("host", "localhost")
      .option("port", 9999)
      .load()

    val sales = socketDF
      .select(
        regexp_extract($"value", regex, 2) as "tpe",
        regexp_extract($"value", regex, 3).cast(IntegerType) as "price",
        regexp_extract($"value", regex, 1) as "material"
      )
      .as[RuneData]

    sales.createOrReplaceTempView("sales")
    val item9850 = spark.sql("SELECT tpe, material, price " +
      "FROM sales " +
      "WHERE price >= 9840 AND price <= 9850 " +
      "GROUP BY tpe, material, price " +
      "HAVING price = 9850")


    val query = item9850
      .writeStream
      .outputMode("complete")
      .format("console")
      .start()


    query.awaitTermination()
    spark.stop()
  }
}

case class RuneData(tpe: String, price: Int, material: String)
```

This code comes from Assignment 5 of the Big Data course at Radboud University, where we had to implement a Spark Structured Streaming application to process streaming data.
In this example, we read streaming data from a socket and extract information about sales of items. We then filter the data to find items that were sold for a price between 9840 and 9850 gp. The results are written to the console. The `awaitTermination` method is called to keep the streaming query running until the user interrupts it. Finally, we stop the Spark session. 
We use the `RuneData` case class to define the schema of the data we are working with.


