package scorch.data.loader

import akka.actor.{Actor, ActorSystem, Props}
import akka.stream._
import akka.stream.scaladsl._
import akka.util.Timeout
import akka.pattern.ask

import scala.concurrent.duration._
import botkop.{numsca => ns}
import botkop.numsca.Tensor
import scorch._
import scorch.autograd.Variable
import scorch.data.loader.NetActor.{Ack, Backward, Complete, Init}
import scorch.data.loader.ParallelLoader.batchSize
import scorch.nn.Infer.Id
import scorch.nn.{Linear, Module}
import scorch.nn.cnn.{Conv2d, MaxPool2d}
import scorch.optim.{Adam, Optimizer}

import scala.collection.immutable
import scala.concurrent.{Await, Future}
import scala.language.postfixOps
import scala.util.{Failure, Success, Try}

class NetActor(net: Module[Id], lossFunction: (Variable, Variable) => Variable)
    extends Actor {
  val optimizer = Adam(net.parameters, lr = 0.01)
  override def receive: Receive = {
    case (x: Variable, y: Variable) =>
      println("received x y")
      val yHat = net(x)
      val loss = lossFunction(yHat, y)
      val cost = loss.data.squeeze()
      println(s"loss = $cost")
      net.zeroGrad()
      loss.backward()

      println(net.parameters.size)

      optimizer.step()
      println("back prop done")
      sender ! Ack

    case _: Init.type =>
      sender ! Ack

    case Complete =>
      println("done")
  }
}

/*
class OptimActor(optimizer: Optimizer) extends Actor {
  override def receive: Receive = {
    case gradients: Seq[Variable] =>
      optimizer.parameters.zip(gradients).foreach { case (p, g) =>
          p.grad.data := g.data
      }
      optimizer.step()
      sender ! optimizer.parameters
  }
}
 */

object NetActor {
  def props(net: Module[Id]) = Props(new NetActor(net, softmaxLoss))
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
                    pad = 1,
                    stride = 1)
  val pool = MaxPool2d(poolSize = 2, stride = 2)
  val numFlatFeatures: Int =
    pool.outputShape(conv.outputShape(inputShape)).tail.product
  def flatten(v: Variable): Variable = v.reshape(batchSize, numFlatFeatures)
  val fc = Linear(numFlatFeatures, numClasses)

  def gradients: Seq[Tensor] = parameters.map(_.grad.data)

  override def forward(x: Variable): Variable =
    x ~> conv ~> relu ~> pool ~> flatten ~> fc ~> relu

}

object ParallelLoader extends App {

  val batchSize = 8
  val loader = new Cifar10DataLoader(miniBatchSize = batchSize,
                                     mode = "train",
                                     take = Some(80))

  implicit val system: ActorSystem = ActorSystem("scorch")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  implicit val askTimeout: Timeout = Timeout(60 seconds)
  import system.dispatcher

  val net = Net(batchSize)
  val netActor = system.actorOf(NetActor.props(net))

  val source = Source(loader)

  val sink = Sink.actorRefWithAck(netActor, Init, Ack, Complete)

  source.runWith(sink)
}

object FlatLoader extends App {
  import scala.concurrent.ExecutionContext.Implicits.global

  val parallelism = 4

  val batchSize = 16

  val loader = new Cifar10DataLoader(miniBatchSize = batchSize,
                                     mode = "train",
                                     take = None)

  val base = Net(batchSize)
  val workers = Seq.fill(parallelism)(Net(batchSize))
  val optimizer = Adam(base.parameters, lr = 0.001)

  def add(as: Seq[Tensor], bs: Seq[Tensor]): Seq[Tensor] =
    as.zip(bs).map { case (a, b) => a + b }

  def pass(iteration: Int, worker: Net, x: Variable, y: Variable): Unit = {
    println("pass")
    val yHat = worker(x)
    val loss = softmaxLoss(yHat, y)
    println(s"iteration: $iteration loss = ${loss.data.squeeze()}")
    loss.backward()
  }

  def updateBaseGradients(allGradients: Seq[Seq[Tensor]]): Unit = {
    val sums = allGradients.fold(base.gradients) {
      case (a, b) => add(a, b)
    }
    val means = sums.map(_ / parallelism)
    base.gradients.zip(means).foreach {
      case (bg, m) =>
        bg := m
    }
  }

  loader.zipWithIndex
    .sliding(parallelism, parallelism)
    .map(_.toList)
    .foreach { pb: List[((Variable, Variable), Int)] =>
      base.zeroGrad()

      val fs: Seq[Future[Seq[Tensor]]] = workers
        .zip(pb)
        .map {
          case (worker, ((x, y), ix)) =>
            Future {
              worker.zeroGrad()
              pass(ix, worker, x, y)
              worker.gradients
            }
        }

      val results = Future.sequence(fs)

      results.onComplete {
        case Success(allGradients) =>
          /*
          val sums = allGradients.fold(base.gradients) {
            case (a, b) => add(a, b)
          }
          val means = sums.map(_ / parallelism)
          base.gradients.zip(means).foreach {
            case (bg, m) =>
              bg := m
          }
           */
          updateBaseGradients(allGradients)

          optimizer.step()

          workers.foreach { w =>
            base.parameters.zip(w.parameters).foreach {
              case (bp, wp) =>
                wp.data := bp.data
            }
          }
          println("step")

        case Failure(ex) =>
          throw new Exception(ex)
      }

      Await.ready(results, 20 seconds)
    }
}

case class ParallelModule(modules: Seq[Module[Id]]) {}

object ParallelModule {

  case class ParallelModuleFunction(x: Variable,
                                    baseModule: Module[Id],
                                    workerModules: Seq[Module[Id]],
                                    timeOut: Duration = Duration.Inf)
      extends scorch.autograd.Function {
    import ns._

    val parallelism: Int = workerModules.length
    val batchSize: Int = x.shape.head

    def scatter(v: Variable): Seq[Variable] =
      (0 until batchSize)
        .sliding(parallelism, parallelism)
        .map(s => (s.head, s.last))
        .map {
          case (first, last) =>
            Variable(v.data(first :> last))
        }
        .toSeq

    // set parameters of all workers to parameters of base module
    // todo: zeroGrad workers or also set gradients?
    workerModules.foreach { wm =>
      wm.parameters.zip(baseModule.parameters).foreach {
        case (wp, bp) =>
          wp.data := bp.data
      }
    }

    val xs: Seq[Variable] = scatter(x)
    val fs: Seq[Future[Variable]] = xs.zip(workerModules).map {
      case (v, worker) =>
        Future(worker(v))
    }
    val results: Seq[Variable] = Await.result(Future.sequence(fs), timeOut)

    override def forward(): Variable = {
      Variable(ns.concatenate(results.map(_.data)), Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      val gs = scatter(gradOutput)
      val fs = results.zip(gs).map {
        case (v, g) =>
          Future(v.backward(g))
      }
      Await.result(Future.sequence(fs), timeOut)

      // collect gradients and back prop
      val allGradients: Seq[Seq[Variable]] = workerModules.map { wm =>
        wm.parameters.map(_.grad)
      }

    }
  }

}
