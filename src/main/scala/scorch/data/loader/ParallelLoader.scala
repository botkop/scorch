package scorch.data.loader

import akka.Done
import akka.actor.{Actor, ActorSystem, Props}
import akka.stream._
import akka.stream.scaladsl._
import akka.util.Timeout
import akka.pattern.ask

import scala.concurrent.duration._
import botkop.numsca.Tensor
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.NetActor.{Ack, Backward, Complete, Init}
import scorch.nn.Infer.Id
import scorch.nn.{Linear, Module}
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch.optim.{Adam, Optimizer}

import scala.concurrent.Future
import scala.language.postfixOps

class NetActor(net: Module[Id]) extends Actor {

  val optimizer = Adam(net.parameters, lr = 0.01)
  override def receive: Receive = {
    case (x: Variable, y: Variable) =>
      println("received x y")
      val yHat = net(x)
      val loss = softmaxLoss(yHat, y)
      val cost = loss.data.squeeze()
      println(s"loss = $cost")
      optimizer.zeroGrad()
      loss.backward()
      optimizer.step()
      println("back prop done")
      sender ! Ack

    case _: Init.type =>
      sender ! Ack

    case Complete =>
      println("done")

  }
}

object NetActor {
  def props(net: Module[Id]) = Props(new NetActor(net))
  case class Backward(loss: Variable)

  case object Init
  case object Ack
  case object Complete
}

case class Net(batchSize: Int) extends Module {
  val numChannels = 3
  val imageSize = 32
  val numClasses = 10
  val inputShape = List(batchSize, numChannels, imageSize, imageSize)
  val conv = Conv2d(numChannels = 3,
                    numFilters = 4,
                    filterSize = 5,
                    weightScale = 1e-3,
                    stride = 1,
                    pad = 1)
  val pool = MaxPool2d(poolSize = 2, stride = 2)
  val numFlatFeatures: Int =
    pool.outputShape(conv.outputShape(inputShape)).tail.product
  def flatten(v: Variable): Variable = v.reshape(batchSize, numFlatFeatures)
  val fc = Linear(numFlatFeatures, numClasses)
  val fc1 = Linear(3 * 32 * 32, numClasses)

  override def forward(x: Variable): Variable =
    x ~> conv ~> relu ~> pool ~> flatten ~> fc ~> relu
}

object ParallelLoader extends App {

  val batchSize = 8
  val loader = new Cifar10DataLoader(miniBatchSize = batchSize, mode = "train", take = Some(80))

  implicit val system: ActorSystem = ActorSystem("scorch")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  implicit val askTimeout: Timeout = Timeout(60 seconds)
  import system.dispatcher

  val net = Net(batchSize)
  val netActor = system.actorOf(NetActor.props(net))

  val source = Source(loader)
    .mapAsync(4) {
      case (x, y) =>
        Future(Variable(x.reshape(batchSize, 3, 32, 32)), Variable(y))
    }
    /*
    .runForeach {
      case (x, y) =>
        (netActor ? (x, y)).mapTo[Double].foreach(println)
    }
    */

  val sink = Sink.actorRefWithAck(netActor, Init, Ack, Complete)

  source.runWith(sink)

}
