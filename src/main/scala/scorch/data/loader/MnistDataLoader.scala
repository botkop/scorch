package scorch.data.loader

import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream

import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable

import scala.io.Source
import scala.util.Random

class MnistDataLoader(mode: String,
                      miniBatchSize: Int,
                      take: Option[Int] = None,
                      seed: Long = 231)
    extends DataLoader
    with LazyLogging {

  val file: String = mode match {
    case "train" => "data/mnist/mnist_train.csv.gz"
    case "dev" => "data/mnist/mnist_test.csv.gz"
  }

  val numEntries: Int =
    Source.fromInputStream(gzis(file)).getLines().length

  override val numSamples: Int = take match {
    case Some(n) => math.min(n, numEntries)
    case None => numEntries
  }

  override val numBatches: Int =
    (numSamples / miniBatchSize) +
      (if (numSamples % miniBatchSize == 0) 0 else 1)

  override def iterator: Iterator[(Variable, Variable)] =
    new Random(seed)
      .shuffle(
        Source
          .fromInputStream(gzis(file))
          .getLines()
      )
      .take(take.getOrElse(numSamples))
      .sliding(miniBatchSize, miniBatchSize)
      .map { lines =>
        val (xData, yData) = lines
          .foldLeft(List.empty[Float], List.empty[Float]) {
            case ((xs, ys), line) =>
              val tokens = line.split(",")
              val (y, x) =
                (tokens.head.toFloat, tokens.tail.map(_.toFloat / 255).toList)
              (x ::: xs, y :: ys)
          }

        val x = Variable(Tensor(xData.toArray).reshape(yData.length, 784))
        val y = Variable(Tensor(yData.toArray).reshape(yData.length, 1))

        (x, y)
      }

  def gzis(fname: String): GZIPInputStream =
    new GZIPInputStream(new BufferedInputStream(new FileInputStream(fname)))

}
