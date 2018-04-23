package scorch.nn

import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable
import scorch.nn.Infer.Id

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration

case class Parallelize(module: Module[Id],
                       parallelism: Int,
                       timeOut: Duration = Duration.Inf)
    extends Module {

  import Parallelize._
  override def forward(x: Variable): Variable =
    ParallelizeFunction(x, module, parallelism, timeOut).forward()
}

object Parallelize {

  case class ParallelizeFunction(x: Variable,
                                 module: Module[Id],
                                 parallelism: Int,
                                 timeOut: Duration = Duration.Inf)
      extends scorch.autograd.Function
      with LazyLogging {
    import ns._
    import scala.concurrent.ExecutionContext.Implicits.global

    val batchSize: Int = x.shape.head
    val chunkSize: Int = batchSize / parallelism

    val fromTos: Seq[(Int, Int)] = (0 until batchSize)
      .sliding(chunkSize, chunkSize)
      .map(s => (s.head, s.last + 1))
      .toSeq

    val fs: Seq[Future[(Variable, Variable)]] = fromTos.map {
      case (first, last) =>
        Future {
          val cx = Variable(x.data(first :> last))
          (cx, module(cx))
        }
    }

    lazy val activations: Seq[(Variable, Variable)] =
      Await.result(Future.sequence(fs), timeOut)
    lazy val xs: Seq[Variable] = activations.map(_._1)
    lazy val predictions: Seq[Variable] = activations.map(_._2)

    override def forward(): Variable =
      Variable(ns.concatenate(predictions.map(_.data)), Some(this))

    override def backward(gradOutput: Variable): Unit = {
      val fs = predictions.zip(fromTos).map {
        case (v, (first, last)) =>
          Future {
            val g = Variable(gradOutput.data(first :> last))
            v.backward(g)
          }
      }
      Await.result(Future.sequence(fs), timeOut)

      val gradient = Variable(ns.concatenate(xs.map(_.grad.data)))
      x.backward(gradient)
    }
  }
}
