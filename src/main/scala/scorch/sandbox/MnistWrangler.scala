package scorch.sandbox

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.data.loader.MnistDataLoader
import scorch.nn.{Linear, Module}
import scorch.optim.SGD
import scorch._

object MnistWrangler extends App {

  val batchSize = 1024
  val lr = 0.03

  case class Net() extends Module {

    val fc1 = Linear(28 * 28, 50)
    val fc2 = Linear(50, 20)
    val fc3 = Linear(20, 10)

    override def forward(x: Variable): Variable =
      x ~> fc1 ~> relu ~> fc2 ~> relu ~> fc3 ~> relu
  }

  val net = Net()
  val trainingSet = new MnistDataLoader("train", batchSize)
  val devSet = new MnistDataLoader("validate", batchSize)
  val optimizer = SGD(net.parameters, lr)

  for (epoch <- 1 to 100) {

    var avgLoss = 0.0
    var avgAccuracy = 0.0
    var count = 0
    val start = System.currentTimeMillis()

    trainingSet.foreach {
      case (x, y) =>
        count += 1

        net.zeroGrad()
        val output = net(x)

        val accuracy = getAccuracy(output, y)
        avgAccuracy += accuracy

        val loss = softmaxLoss(output, y)
        avgLoss += loss.data.squeeze()

        loss.backward()
        optimizer.step()
    }
    val stop = System.currentTimeMillis()
    println(
      s"training: $epoch: loss: ${avgLoss / count} accuracy: ${avgAccuracy / count} time: ${stop - start}")

    evaluate(net, epoch)
  }

  def evaluate(model: Module, epoch: Int): Unit = {
    model.eval()

    var avgLoss = 0.0
    var avgAccuracy = 0.0
    var count = 0

    devSet.foreach {
      case (x, y) =>
        count += 1
        val output = net(x)
        val guessed = ns.argmax(output.data, axis = 1)
        val accuracy = ns.sum(guessed == y.data) / batchSize
        avgAccuracy += accuracy
        val loss = softmaxLoss(output, y)
        avgLoss += loss.data.squeeze()
    }
    println(
      s"testing:  $epoch: loss: ${avgLoss / count} accuracy: ${avgAccuracy / count}")

    model.train()
  }

  def getAccuracy(yHat: Variable, y: Variable): Double = {
    val guessed = ns.argmax(yHat.data, axis = 1)
    ns.mean(y.data == guessed).squeeze()
  }
}
