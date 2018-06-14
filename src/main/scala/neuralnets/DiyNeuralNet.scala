package neuralnets

import breeze.linalg._
import breeze.stats.distributions._

import scala.annotation.tailrec


object DiyNeuralNet extends App {

  import Neuron.{runGradientDescent, prediction}
  import Hydration.{data, weights, labels}

  val optimizedWeights: DenseVector[Double] = runGradientDescent(data, labels, alpha = 0d)
  println(s"Optimized weights: $optimizedWeights")
  val predictions: List[Double] = prediction(weights, data).toArray.toList
  println(s"\nPredictions:\n$predictions\n")
}

object Hydration {
  val data: DenseMatrix[Double] =
    DenseMatrix(
      (1.0, 1.0),
      (0.0, 1.0),
      (1.0, 0.0),
      (0.0, 0.0)
    )
  val weights: DenseVector[Double] = DenseVector(1.0, 1.0)
  val labels: DenseVector[Double] = DenseVector(1.0, 1.0, 0.0, 0.0) // learning to just look at the second element
}


object Neuron {

  /**
    * J = 0.5 * [Xw-y]^2 + (alpha * w^2)
    *
    * @param weights model params; I x 1 vector
    * @param data to be labeled by model; N x I matrix
    * @param labels truth; N x 1 vector
    * @param alpha constant for L2 regularization
    * @return
    */
  def j(weights: DenseVector[Double],
        data: DenseMatrix[Double],
        labels: DenseVector[Double],
        alpha: Double): Double =
  {
    val error: DenseVector[Double] = data * weights - labels
    val regularizationPenalty: Double = alpha * weights dot weights
    0.5 * error dot error + regularizationPenalty
  }


  /**
    * gradJ = 0.5 * 2 * X.T(Xw - y) = X.T(Xw-y) + (Alpha * w)
    *
    * @param weights model params; I x 1 vector
    * @param data to be labeled by model; N x I matrix
    * @param labels truth; N x 1 vector
    * @param alpha constant for L2 regularization
    * @return
    */
  def gradJ(weights: DenseVector[Double],
            data: DenseMatrix[Double],
            labels: DenseVector[Double],
            alpha: Double): DenseVector[Double] =
  {
    val error: DenseVector[Double] = data * weights - labels
    (data.t * error) + (alpha * weights)
  }


  // no bias term
  def runGradientDescent(data: DenseMatrix[Double],
                         labels: DenseVector[Double],
                         alpha: Double,
                         learningRate: Double = 1e-2): DenseVector[Double] = {
    val standardNormal: Gaussian = Gaussian(mu = 0.0, sigma = 1.0)
    val initialWeights: DenseVector[Double] = DenseVector(standardNormal.samples.take(data.cols).toArray)
    val prevJ: Double = j(initialWeights, data, labels, alpha)
    gradientDescent(initialWeights, data, labels, alpha, learningRate, prevJ = prevJ)
  }


  @tailrec
  def gradientDescent(weights: DenseVector[Double],
                      data: DenseMatrix[Double],
                      labels: DenseVector[Double],
                      alpha: Double,
                      learningRate: Double,
                      prevJ: Double,
                      diff: Double = Double.MaxValue,
                      count: Double = 0d,
                      delta: Double = 1e-6): DenseVector[Double] = {
    if (diff < delta) {
      println(s"\nConverged! Step $count: \t Final Cost: $prevJ")
//      println(s"\tFinal accuracy: ${accuracy(weights, data, labels)}")
      weights
    }
    else {
      val update: DenseVector[Double] = learningRate * gradJ(weights, data, labels, alpha)
      val newWeights: DenseVector[Double] = weights - update
      val newJ = j(newWeights, data, labels, alpha)
      val newDiff = prevJ - newJ
      if (count % 1 == 0)
        println(s"Step $count: \t $newJ $diff")
      gradientDescent(newWeights, data, labels, alpha, learningRate,
        prevJ = newJ, diff = newDiff, count = count + 1)
    }
  }

  def prediction(weights: DenseVector[Double],
                 data: DenseMatrix[Double]): DenseVector[Double] = {
    data * weights
  }

  // note that this only makes sense for classification problems, not regression
//  def accuracy(weights: DenseVector[Double],
//               data: DenseMatrix[Double],
//               labels: DenseVector[Double]): Double = {
//    val predictions: Array[Double] = prediction(weights, data).toArray
//    println(s"Data:\n$data\n")
//    println(s"Predictions:\n$predictions\n")
//    val labelsArr: Array[Double] = labels.toArray
//    require(predictions.length == labelsArr.length)
//
//    val predictionAccuracy: Seq[Double] =
//      for {
//        (predicted, label) <- predictions zip labelsArr
//      } yield {
//        if (predicted == label) 1.0 else 0.0
//      }
//
//    predictionAccuracy.sum / labels.length
//  }
}
