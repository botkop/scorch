package scorch.sandbox.cnn

import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.{Cifar10DataLoader, DataLoader}
import scorch.nn._
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch.optim.{Adam, Optimizer}

import scala.collection.parallel.CollectionConverters._

object LeNet5 extends App with LazyLogging {

  case class Net() extends Module {

    val c1 = Conv2d(numChannels = 3,
                    numFilters = 6,
                    filterSize = 3,
                    weightScale = 1e-3,
                    pad = 1,
                    stride = 1)

    val p2 = MaxPool2d(poolSize = 2, stride = 2)

    val c3 = Conv2d(numChannels = 6,
                    numFilters = 16,
                    filterSize = 5,
                    weightScale = 1e-3,
                    pad = 1,
                    stride = 1)

    val p4 = MaxPool2d(poolSize = 2, stride = 2)

    val c5 = Conv2d(numChannels = 16,
                    numFilters = 120,
                    filterSize = 5,
                    weightScale = 1e-3,
                    stride = 1,
                    pad = 1)

    val numFlatFeatures: Int =
      c5.outputShape(p4.outputShape(
          c3.outputShape(p2.outputShape(c1.outputShape(List(-1, 3, 32, 32))))))
        .tail
        .product

    def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)

    println(numFlatFeatures)

    val f6 = Linear(numFlatFeatures, 84)
    val f7 = Linear(84, 10)

    override def forward(xf: Variable): Variable = {
      val x = Variable(xf.data.reshape(-1, 3, 32, 32))
      x ~>
        c1 ~> relu ~> p2 ~>
        c3 ~> relu ~> p4 ~>
        c5 ~> relu ~> flatten ~>
        f6 ~> relu ~> f7
    }
  }

  case class FcNet() extends Module {

    val fc1 = Linear(784, 100)
    val fc2 = Linear(100, 10)

    val bn = BatchNorm(100)
    val drop = Dropout()

    override def forward(x: Variable): Variable =
      x ~> fc1 ~> relu ~> drop ~> fc2 ~> relu
  }

  case class CNN1() extends Module {
    val c1 = Conv2d(numChannels = 3,
                    numFilters = 6,
                    filterSize = 3,
                    weightScale = 1e-3,
                    pad = 1,
                    stride = 1)

    val p1 = MaxPool2d(poolSize = 2, stride = 2)
    val inputShape = List(-1, 3, 32, 32)
    val numFlatFeatures: Int =
      // c1.outputShape(inputShape).tail.product
    p1.outputShape(c1.outputShape(inputShape)).tail.product
    def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)
    val fc1 = Linear(numFlatFeatures, 10)
    override def forward(x: Variable): Variable =
//      x ~> c1 ~> relu ~> flatten ~> fc1 ~> relu
     x ~> c1 ~> relu ~> p1 ~> flatten ~> fc1 ~> relu
  }

  case class FcNet2() extends Module {

    val fc1 = Linear(3 * 32 * 32, 100)
    val fc2 = Linear(100, 10)

    val drop = Dropout()

    override def forward(x: Variable): Variable =
      x ~> fc1 ~> relu ~> drop ~> fc2 ~> relu
  }

  // val batchSize = 128
  val batchSize = 8
  val printEvery = 5

  // val net = Net().par()
  // val net = FcNet().par()
  //val net = CNN1().par()
  // val net = FcNet2().par()
  val net = Net()

  // set in training mode for drop out / batch norm
  net.train()

  val optimizer = Adam(net.parameters, lr = 1e-3)

  /*
  def loader =
    new MnistDataLoader(mode = "train",
                        miniBatchSize = batchSize,
                        seed = System.currentTimeMillis())

  def testLoader = new MnistDataLoader(mode = "dev", miniBatchSize = batchSize)
   */

  def loader =
    new Cifar10DataLoader(mode = "train", miniBatchSize = batchSize, seed = 231, take = Some(80))
    // new Cifar10DataLoader(mode = "train", miniBatchSize = batchSize, seed = System.currentTimeMillis())
  def testLoader =
    new Cifar10DataLoader(mode = "dev", miniBatchSize = batchSize)

  loop(net, optimizer, loader, testLoader, batchSize, printEvery)

  def loop(model: Module,
           optimizer: Optimizer,
           trainLoader: => DataLoader,
           testLoader: => DataLoader,
           batchSize: Int,
           printEvery: Int): Unit = {

    for (epoch <- 1 to 100) {

      var avgLoss = 0.0

      var epochAvgAccuracy = 0.0
      var epochAvgLoss = 0.0
      var iteration = 0
      var iterationStart = System.currentTimeMillis()

      logger.info(s"starting epoch $epoch")
      val epochStart = System.currentTimeMillis()

      loader.foreach {
        case (x, y) =>
          iteration += 1

          optimizer.zeroGrad()
          val yHat = model(x)

          val loss = softmaxLoss(yHat, y)
          avgLoss += loss.data.squeeze()

          if (iteration % printEvery == 0) {
            avgLoss /= printEvery
            val guessed = ns.argmax(yHat.data, axis = 1)
            val accuracy = ns.sum(y.data == guessed) / batchSize
            val iterationEnd = System.currentTimeMillis()
            logger.info(s"$epoch:$iteration: loss: $avgLoss accuracy: $accuracy duration: ${iterationEnd - iterationStart} ms.")

            epochAvgAccuracy += accuracy
            epochAvgLoss += avgLoss

            avgLoss = 0.0
            iterationStart = iterationEnd
          }

          loss.backward()
          optimizer.step()
      }

      val epochDuration = System.currentTimeMillis() - epochStart
      logger.info(s"epoch took $epochDuration ms")
      // evaluateModel(model, testLoader)
      logger.info(s"training accuracy = ${epochAvgAccuracy / (iteration / printEvery)}")
      logger.info(s"epoch loss = ${epochAvgLoss / (iteration / printEvery)}")
      logger.info("================")

    }
  }

  def evaluateModel(model: Module, testLoader: DataLoader): Unit = {
    // set in evaluation mode
    model.eval()
    val avgAccuracy =
      testLoader.par.map {
        case (x, y) =>
          val yHat = model(x)
          val guessed = ns.argmax(yHat.data, axis = 1)
          ns.sum(y.data == guessed) / x.shape.head
      }.sum / testLoader.size
    logger.info(s"evaluation accuracy = $avgAccuracy")

    // set back to training mode
    model.train()
  }

}
