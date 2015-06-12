/**
 * Created by rui on 09/06/15.
 */

package com.datascience

import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object RunKMeans {

  def main(args: Array[String]): Unit = {

    // Reads Dataset
    val sc = new SparkContext(new SparkConf().setAppName("RunKMeans").setMaster(args(0)))
    //val rawData = sc.textFile("/home/rui/Find-Worst-Outlier/kmeans-outlier/datasets/mouse_classified.csv")

    //anomalies(rawData)
  }

  def anomalies(rawData: RDD[String]) = {
    val parseFunction = buildCategoricalAndLabelFunction(rawData)
    parseFunction.foreach(println)

    /*val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
    originalAndData.foreach(println)

    val data = originalAndData.values
    val normalizeFunction = buildNormalizationFunction(data)
    val anomalyDetector = buildAnomalyDetector(data, normalizeFunction)
    val anomalies = originalAndData.filter {
      case (original, datum) => anomalyDetector(datum)
    }.keys
    anomalies.take(10).foreach(println)*/
  }

  def buildCategoricalAndLabelFunction(rawData: RDD[String]): RDD[(String, Vector)] = {

    return rawData.map { line =>
      val buffer = line.split(' ').toBuffer
      val label = buffer.remove(buffer.length - 1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }
  }

  def buildNormalizationFunction(data: RDD[Vector]): (Vector => Vector) = {
    val dataAsArray = data.map(_.toArray)
    val numCols = dataAsArray.first().length
    val n = dataAsArray.count()
    val sums = dataAsArray.reduce(
      (a, b) => a.zip(b).map(t => t._1 + t._2))
    val sumSquares = dataAsArray.fold(
      new Array[Double](numCols)
    )(
        (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2)
      )
    val stdevs = sumSquares.zip(sums).map {
      case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
    }
    val means = sums.map(_ / n)

    (datum: Vector) => {
      val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
        (value, mean, stdev) =>
          if (stdev <= 0)  (value - mean) else  (value - mean) / stdev
      )
      Vectors.dense(normalizedArray)
    }
  }

  def buildAnomalyDetector(data: RDD[Vector],
                           normalizeFunction: (Vector => Vector)): (Vector => Boolean) = {
    val normalizedData = data.map(normalizeFunction)
    normalizedData.cache()

    val kmeans = new KMeans()
    kmeans.setK(150)
    kmeans.setRuns(10)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(normalizedData)

    normalizedData.unpersist()

    val distances = normalizedData.map(datum => distToCentroid(datum, model))
    val threshold = distances.top(100).last

    (datum: Vector) => distToCentroid(normalizeFunction(datum), model) > threshold
  }

  def distToCentroid(datum: Vector, model: KMeansModel) = {
    val cluster = model.predict(datum)
    val centroid = model.clusterCenters(cluster)
    distance(centroid, datum)
  }

  def distance(a: Vector, b: Vector) =
    math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)
}
