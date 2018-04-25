package scorch.sandbox.cnn

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.nn._
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch._
import scorch.data.loader.{Cifar10DataLoader, DataLoader, MnistDataLoader}
import scorch.optim.{Adam, Nesterov, Optimizer}

object LeNet5 extends App {

  case class Net() extends Module {

    val c1 = Conv2d(numChannels = 1,
                    numFilters = 6,
                    filterSize = 5,
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
          c3.outputShape(p2.outputShape(c1.outputShape(List(-1, 1, 28, 28))))))
        .tail
        .product

    def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)

    println(numFlatFeatures)

    val f6 = Linear(numFlatFeatures, 84)
    val f7 = Linear(84, 10)

    override def forward(xf: Variable): Variable = {
      val x = Variable(xf.data.reshape(-1, 1, 28, 28))
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
      p1.outputShape(c1.outputShape(inputShape)).tail.product
    def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)
    val fc1 = Linear(numFlatFeatures, 10)
    override def forward(x: Variable): Variable =
      x ~> c1 ~> relu ~> p1 ~> flatten ~> fc1 ~> relu
  }

  case class FcNet2() extends Module {

    val fc1 = Linear(3 * 32 * 32, 100)
    val fc2 = Linear(100, 10)

    val drop = Dropout()

    override def forward(x: Variable): Variable =
      x ~> fc1 ~> relu ~> drop ~> fc2 ~> relu
  }

  val batchSize = 128
  val printEvery = 100

  // val net = Net().par()
  // val net = FcNet().par()
  // val net = CNN1().par(8)
  val net = FcNet2().par()

  val optimizer = Adam(net.parameters, lr = 1e-3)

  /*
  def loader =
    new MnistDataLoader(mode = "train",
                        miniBatchSize = batchSize,
                        seed = System.currentTimeMillis())

  def testLoader = new MnistDataLoader(mode = "dev", miniBatchSize = batchSize)
   */

  def loader =
    new Cifar10DataLoader(mode = "train",
                          miniBatchSize = batchSize,
                          seed = System.currentTimeMillis(),
                          tailShape = Seq(3 * 32 * 32))
  def testLoader =
    new Cifar10DataLoader(mode = "dev",
                          miniBatchSize = batchSize,
                          tailShape = Seq(3 * 32 * 32))

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

      println(s"starting epoch $epoch")
      var epochStart = System.currentTimeMillis()

      trainLoader.foreach {
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
            println(s"$epoch:$iteration: loss: $avgLoss accuracy: $accuracy")

            epochAvgAccuracy += accuracy
            epochAvgLoss += avgLoss

            avgLoss = 0.0
          }

          loss.backward()
          optimizer.step()
      }

      val epochDuration = System.currentTimeMillis() - epochStart
      println(s"epoch took $epochDuration ms")
      evaluateModel(model, testLoader)
      println(
        s"training accuracy = ${epochAvgAccuracy / (iteration / printEvery)}")
      println(s"epoch loss = ${epochAvgLoss / (iteration / printEvery)}")
      println("================")

    }
  }

  def evaluateModel(model: Module, testLoader: DataLoader): Unit = {

    var avgAccuracy = 0.0
    testLoader.par.foreach {
      case (x, y) =>
        val yHat = model(x)
        val guessed = ns.argmax(yHat.data, axis = 1)
        val accuracy = ns.sum(y.data == guessed) / x.shape.head
        avgAccuracy += accuracy
    }
    avgAccuracy /= testLoader.size
    println(s"evaluation accuracy = $avgAccuracy")
  }

}
