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
                                 baseModule: Module[Id],
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

    val chunkedXs: Seq[Variable] = fromTos.map {
      case (first, last) => Variable(x.data(first :> last))
    }

    val fs: Seq[Future[Variable]] = chunkedXs.map { cx =>
      Future {
        baseModule(cx)
      }
    }

    val activations: Seq[Variable] = Await.result(Future.sequence(fs), timeOut)

    override def forward(): Variable =
      Variable(ns.concatenate(activations.map(_.data)), Some(this))

    override def backward(gradOutput: Variable): Unit = {
      val fs = activations.zip(fromTos).map {
        case (v, (first, last)) =>
          Future {
            val g = Variable(gradOutput.data(first :> last))
            // todo: thread safe ?
            v.backward(g)
          }
      }
      Await.result(Future.sequence(fs), timeOut)

      val gradient = Variable(ns.concatenate(chunkedXs.map(_.grad.data)))
      x.backward(gradient)
    }
  }
}

