package scorch.sandbox.cnn

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.nn.{Linear, Module, ParallelizeModule}
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch._
import scorch.data.loader.MnistDataLoader
import scorch.optim.{Adam, Nesterov}

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
        c5.outputShape(p4.outputShape(c3.outputShape(p2.outputShape(c1.outputShape(List(-1, 1, 28, 28))))))
      .tail
      .product

    def flatten(v: Variable): Variable = v.reshape(-1, numFlatFeatures)

    println(numFlatFeatures)

    val f6 = Linear(numFlatFeatures, 84)
    val f7 = Linear(84, 10)

    override def forward(x: Variable): Variable =
      x ~>
        c1 ~> relu ~> p2 ~>
        c3 ~> relu ~> p4 ~>
        c5 ~> relu ~> flatten ~>
        f6 ~> relu ~> f7
  }

  val net = Net().par()
  val optimizer = Nesterov(net.parameters, lr = 2e-2)

  val batchSize = 16

  for (epoch <- 1 to 16) {
    val loader = new MnistDataLoader(mode = "train", miniBatchSize = batchSize)
    loader.zipWithIndex.foreach {
      case ((xf, y), iteration) =>
        val x = Variable(xf.data.reshape(-1, 1, 28, 28))

        optimizer.zeroGrad()
        val yHat = net(x)

        val loss = softmaxLoss(yHat, y)
        val guessed = ns.argmax(yHat.data, axis = 1)
        val accuracy = ns.sum(y.data == guessed) / batchSize
        println(
          s"$epoch:$iteration: loss: ${loss.data.squeeze()} accuracy: $accuracy")

        loss.backward()
        optimizer.step()
    }
  }
}
