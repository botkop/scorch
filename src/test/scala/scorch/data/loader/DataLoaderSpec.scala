package scorch.data.loader

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}
import scorch._
import scorch.autograd.Variable
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch.nn.{Linear, Module}
import scorch.optim.Adam

import scala.concurrent.duration._
import scala.language.postfixOps

class DataLoaderSpec extends FlatSpec with Matchers {

  "A cifar-10 loader" should "load data" in {

    val miniBatchSize = 8
    val take = 10

    val loader =
      new Cifar10DataLoader(miniBatchSize = miniBatchSize,
                            mode = "train",
                            take = Some(take * miniBatchSize)).toSeq

    assert(loader.size == take)

    loader.foreach {
      case (x, y) =>
        assert(x.shape.head == miniBatchSize)
        // assert(x.shape(1) == 3 * 32 * 32)
        assert(y.shape.head == miniBatchSize)
        assert(y.shape(1) == 1)
    }
  }

  it should "feed a network" in {

    val batchSize = 16
    val numBatches = 2

    val (numChannels, imageSize) = (3, 32)
    val inputShape = List(batchSize, numChannels, imageSize, imageSize)
    val numClasses = 10

    case class Net() extends Module {
      val conv = Conv2d(numChannels = 3,
                        numFilters = 4,
                        filterSize = 5,
                        weightScale = 1e-3,
                        stride = 1,
                        pad = 1)
      val pool = MaxPool2d(poolSize = 2, stride = 2)
      val numFlatFeatures: Int =
        pool.outputShape(conv.outputShape(inputShape)).tail.product
      // def flatten(v: Variable): Variable = v.reshape(batchSize, numFlatFeatures)
      def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)
      val fc = Linear(numFlatFeatures, numClasses)

      override def forward(x: Variable): Variable =
        x ~> conv ~> relu ~> pool ~> flatten ~> fc ~> relu
    }

    val net = Net()

    val pNet = scorch.nn.Parallelize(net, 4, 20 seconds)

    val optimizer = Adam(net.parameters, lr = 0.001)
    val loader = new Cifar10DataLoader(miniBatchSize = batchSize,
                                       mode = "train",
                                       take = Some(numBatches * batchSize))

    val seq = loader.toSeq

    for (epoch <- 0 to 10) {
      seq.zipWithIndex
        .foreach {
          case ((x, y), i) =>
            optimizer.zeroGrad()
            val yHat = pNet(x)
            val loss = softmaxLoss(yHat, y)

            val guessed = ns.argmax(yHat.data, axis = 1)
            val accuracy = ns.sum(y.data == guessed) / batchSize
            println(
              s"$epoch:$i: loss: ${loss.data.squeeze()} accuracy: $accuracy")

            loss.backward()
            optimizer.step()
        }
    }
  }
}
