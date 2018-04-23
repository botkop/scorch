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

    val fs: Seq[Future[Variable]] = fromTos.map {
      case (first, last) =>
        Future {
          val v = Variable(x.data(first :> last))
          logger.info(s"shape of this chunk: ${v.shape}")
          baseModule(v)
        }
    }

    val activations: Seq[Variable] = Await.result(Future.sequence(fs), timeOut)

    override def forward(): Variable =
      Variable(ns.concatenate(activations.map(_.data)), Some(this))

    override def backward(gradOutput: Variable): Unit = {
      /*
      val fs = activations.zip(fromTos).map {
        case (v, (first, last)) =>
          Future {
            val g = Variable(gradOutput.data(first :> last))
            logger.info(s"shape of this gradient: ${g.shape}")
            // todo: not thread safe
            v.backward(g)
          }
      }
      Await.result(Future.sequence(fs), timeOut)
      */

      activations.zip(fromTos).foreach {
        case (v, (first, last)) =>
          val g = Variable(gradOutput.data(first :> last))
          logger.info(s"shape of this gradient: ${g.shape}")
          v.backward(g)
      }

    }
  }
}
/*

case class Parallelize(module: Module[Id],
                       parallelism: Int,
                       timeOut: Duration = Duration.Inf)
    extends Module {

  val workers: Seq[Module[Id]] =
    Seq.fill(parallelism)(module.clone().asInstanceOf[Module[Id]])

  import Parallelize._
  override def forward(x: Variable): Variable =
    ParallelizeFunction(x, module, workers, timeOut).forward()
}

object Parallelize {

  case class ParallelizeFunction(x: Variable,
                                 baseModule: Module[Id],
                                 workerModules: Seq[Module[Id]],
                                 timeOut: Duration = Duration.Inf)
      extends scorch.autograd.Function {
    import ns._
    import scala.concurrent.ExecutionContext.Implicits.global

    val parallelism: Int = workerModules.length
    val batchSize: Int = x.shape.head

    val fromTos: Seq[(Int, Int)] = (0 until batchSize)
      .sliding(parallelism, parallelism)
      .map(s => (s.head, s.last + 1))
      .toSeq

    def scatter(v: Variable): Seq[Variable] =
      fromTos.map {
        case (first, last) =>
          Variable(v.data(first :> last))
      }.toSeq

    val fs: Seq[Future[Variable]] = workerModules.zip(fromTos).map {
      case (worker, (first, last)) =>
        Future {
          /*
          // set parameters of worker to parameters of base module
          // set gradients of worker to gradients of base module (supposedly those are zero)
          worker.parameters.zip(baseModule.parameters).foreach {
            case (wp, bp) =>
              wp.data := bp.data
              wp.grad.data := bp.grad.data
          }

          val v = Variable(x.data(first :> last))
          worker(v)
 */
          val v = Variable(x.data(first :> last))
          baseModule(v)
        }
    }

    val activations: Seq[Variable] = Await.result(Future.sequence(fs), timeOut)

    override def forward(): Variable = {
      Variable(ns.concatenate(activations.map(_.data)), Some(this))
    }

    override def backward(gradOutput: Variable): Unit = {
      val fs = activations.zip(fromTos).map {
        case (v, (first, last)) =>
          Future {
            val g = Variable(gradOutput.data(first :> last))
            v.backward(g)
          }
      }
      Await.result(Future.sequence(fs), timeOut)

      // collect gradients from workers and accumulate in base module
      workerModules.foreach { wm =>
        baseModule.gradients.zip(wm.gradients).foreach {
          case (bg, wg) =>
            bg.data += wg.data
        }
      }
    }
  }
}
 */
