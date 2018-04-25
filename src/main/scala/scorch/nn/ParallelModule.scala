package scorch.nn

import botkop.{numsca => ns}
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable

import scala.collection.parallel.mutable.ParArray

case class ParallelModule(module: Module, parallelism: Int) extends Module {

  import ParallelModule._
  override def forward(x: Variable): Variable =
    ParallelizeFunction(x, module, parallelism).forward()
}

object ParallelModule {

  case class ParallelizeFunction(x: Variable, module: Module, parallelism: Int)
      extends scorch.autograd.Function
      with LazyLogging {
    import ns._

    val batchSize: Int = x.shape.head
    val chunkSize: Int = Math.max(batchSize / parallelism, 1)

    val fromTos: ParArray[(Int, Int)] = (0 until batchSize)
      .sliding(chunkSize, chunkSize)
      .map(s => (s.head, s.last + 1))
      .toArray
      .par

    lazy val (xs, predictions) =
      fromTos.map {
        case (first, last) =>
          val cx = Variable(x.data(first :> last))
          (cx, module(cx))
      }.unzip

    override def forward(): Variable =
      Variable(ns.concatenate(predictions.map(_.data).seq), Some(this))

    override def backward(gradOutput: Variable): Unit = {
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
