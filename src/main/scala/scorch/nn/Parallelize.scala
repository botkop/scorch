package scorch.nn

import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable

import scala.collection.parallel.ParSeq
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}

case class Parallelize(module: Module,
                       parallelism: Int,
                       timeOut: Duration = Duration.Inf)
    extends Module {

  import Parallelize._
  override def forward(x: Variable): Variable =
    ParallelizeFunction(x, module, parallelism, timeOut).forward()
}

object Parallelize {

  case class ParallelizeFunction(x: Variable,
                                 module: Module,
                                 parallelism: Int,
                                 timeOut: Duration = Duration.Inf)
      extends scorch.autograd.Function
      with LazyLogging {
    import ns._

    import scala.concurrent.ExecutionContext.Implicits.global

    val batchSize: Int = x.shape.head
    val chunkSize: Int = batchSize / parallelism

    val fromTos: ParSeq[(Int, Int)] = (0 until batchSize)
      .sliding(chunkSize, chunkSize)
      .map(s => (s.head, s.last + 1))
      .toSeq
      .par

    /*
    val fs: Seq[Future[(Variable, Variable)]] = fromTos.map {
      case (first, last) =>
        Future {
          val cx = Variable(x.data(first :> last))
          (cx, module(cx))
        }
    }

    lazy val (xs: Seq[Variable], predictions: Seq[Variable]) =
      Await.result(Future.sequence(fs), timeOut).unzip
      */

    lazy val (xs, predictions) =
      fromTos.map {
        case (first, last) =>
          val cx = Variable(x.data(first :> last))
          (cx, module(cx))
      }.unzip

    override def forward(): Variable =
      Variable(ns.concatenate(predictions.map(_.data).seq), Some(this))

    override def backward(gradOutput: Variable): Unit = {
      /*
      val fs = predictions.zip(fromTos).map {
        case (v, (first, last)) =>
          Future {
            val g = Variable(gradOutput.data(first :> last))
            v.backward(g)
          }
      }
      Await.result(Future.sequence(fs), timeOut)
      */

      predictions.zip(fromTos).foreach {
        case (v, (first, last)) =>
            val g = Variable(gradOutput.data(first :> last))
            v.backward(g)
      }

      val gradient = Variable(ns.concatenate(xs.map(_.grad.data).seq))
      x.backward(gradient)
    }
  }
}
