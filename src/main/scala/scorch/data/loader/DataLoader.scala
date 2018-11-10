package scorch.data.loader

import scorch.autograd.Variable

trait DataLoader extends scala.collection.immutable.Iterable[(Variable, Variable)] {
  def numSamples: Int
  def numBatches: Int
  def mode: String
}

object DataLoader {
  def instance(dataSet: String,
               mode: String,
               miniBatchSize: Int,
               take: Option[Int] = None): DataLoader =
    dataSet match {
      case "cifar-10" =>
        new Cifar10DataLoader(mode, miniBatchSize, take = take)
      case "mnist" =>
        new MnistDataLoader(mode, miniBatchSize, take)
    }
}

@SerialVersionUID(123L)
case class YX(y: Float, x: Array[Float]) extends Serializable
