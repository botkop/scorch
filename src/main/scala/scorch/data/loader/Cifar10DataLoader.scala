package scorch.data.loader

import java.awt.image.BufferedImage
import java.io._
import javax.imageio.ImageIO

import botkop.numsca
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging

import scala.collection.parallel.mutable.ParArray
import scala.language.postfixOps
import scala.util.Random

class Cifar10DataLoader(mode: String,
                        miniBatchSize: Int,
                        take: Option[Int] = None,
                        seed: Long = 231)
    extends DataLoader
    with LazyLogging {

  import Cifar10DataLoader._

  val folder: String =
    if (mode == "train") "data/cifar-10/train.yx" else "data/cifar-10/dev.yx"

  val files: List[File] = new File(folder).listFiles
    .filter(_.isFile)
    .toList

  override val numSamples: Int = take match {
    case Some(n) => math.min(n, files.length)
    case None => files.length
  }

  override val numBatches: Int =
    (numSamples / miniBatchSize) +
      (if (numSamples % miniBatchSize == 0) 0 else 1)

  override def iterator: Iterator[(Tensor, Tensor)] = {
    val batches: Iterator[List[File]] = new Random(seed)
      .shuffle(files)
      .take(numSamples)
      .grouped(miniBatchSize)

    batches.map { sampleFiles =>
      val batchSize = sampleFiles.length

      // todo: maybe use akka streams here
      val yxs = sampleFiles.par map deserialize

      val xData = yxs flatMap (_.x) toArray
      val yData = yxs map (_.y) toArray

      val x = Tensor(xData).reshape(batchSize, numFeatures)
      val y = Tensor(yData).reshape(batchSize, 1)

      (x, y)
    }
  }

}

object Cifar10DataLoader extends LazyLogging {

  val numFeatures: Int = 32 * 32 * 3

  def computeMeanImage(files: List[File]): Tensor = {
    val mean = numsca.zeros(1, numFeatures)

    files.foreach { f =>
      val lof = readImage(f)
      val t = Tensor(lof).reshape(1, numFeatures)
      mean += t
    }
    mean / files.length
  }

  def readImage(file: File): Array[Float] = {
    val image: BufferedImage = ImageIO.read(file)

    val w = image.getWidth
    val h = image.getHeight
    val div: Float = 255

    val lol  = for {
      i <- 0 until h
      j <- 0 until w
    } yield {
      val pixel = image.getRGB(i, j)
      val red = ((pixel >> 16) & 0xff) / div
      val green = ((pixel >> 8) & 0xff) / div
      val blue = (pixel & 0xff) / div
      Seq(red, green, blue)
    }
    lol.toArray flatten
  }

  def getListOfFiles(dir: String): ParArray[(Float, File)] = {

    val labels = List(
      "airplane",
      "automobile",
      "bird",
      "cat",
      "deer",
      "dog",
      "frog",
      "horse",
      "ship",
      "truck"
    )

    val d = new File(dir)
    d.listFiles
      .filter(_.isFile)
      .map { f =>
        val Array(seq, cat) =
          f.getName.replaceFirst("\\.png", "").split("_")
        (seq.toInt, cat, f)
      }
      .sortBy(_._1)
      .map {
        case (_, cat, file) =>
          (labels.indexOf(cat).toFloat, file)
      }
      .par
  }

  def serializeDataset(outputFolder: String,
                       fileList: ParArray[(Float, File)],
                       meanImage: Tensor): Unit =
    fileList.foreach {
      case (yData, file) =>
        val xData = readImage(file)
        val x = Tensor(xData).reshape(1, numFeatures) - meanImage

        val oos = new ObjectOutputStream(
          new FileOutputStream(s"$outputFolder/${file.getName}.yx"))
        oos.writeObject(YX(yData, x.data.map(_.toFloat)))
        oos.close()
    }

  def deserialize(file: File): YX = {
    val ois = new ObjectInputStream(new FileInputStream(file))
    val yx = ois.readObject.asInstanceOf[YX]
    ois.close()
    yx
  }

  def main(args: Array[String]): Unit = {
    val trainingFileList = getListOfFiles("/Users/koen/projects/nazca/data/cifar-10/train")
    val devFileList = getListOfFiles("/Users/koen/projects/nazca/data/cifar-10/test")

    logger.debug("calculating mean image from training set")
    // cannot use parallel collection for computing mean!
    val meanImage = computeMeanImage(trainingFileList.toList.map(_._2))

    logger.debug("serializing training set")
    serializeDataset("data/cifar-10/train.yx", trainingFileList, meanImage)

    logger.debug("serializing dev set")
    serializeDataset("data/cifar-10/dev.yx", devFileList, meanImage)
  }

}
