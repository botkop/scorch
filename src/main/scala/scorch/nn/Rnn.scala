package scorch.nn

import botkop.numsca.:>
import botkop.{numsca => ns}
import scorch.autograd._

case class Rnn(h0: Variable, wX: Variable, wH: Variable, b: Variable)
    extends Module(Seq(wX, wH, b)) {

  override def forward(x: Variable): Variable =
    RnnFunction(x, h0, wX, wH, b).forward()

}

object Rnn {
  def apply(h0: Variable, d: Int): Rnn = {
    val List(n, h) = h0.shape
    val wX = Variable(ns.randn(d, h) / math.sqrt(d))
    val wH = Variable(ns.randn(h, h) / math.sqrt(h))
    val b = Variable(ns.zeros(h))
    Rnn(h0, wX, wH, b)
  }
}

case class RnnFunction(x: Variable,
                       h0: Variable,
                       wX: Variable,
                       wH: Variable,
                       b: Variable)
    extends Function {
  import RnnFunction._

  val List(n, t, d) = x.shape
  val List(_, h) = h0.shape

  lazy val vs: List[Variable] = (0 until t).foldLeft(List.empty[Variable]) {
    case (acc, i) =>
      val v = Variable(x.data(:>, i, :>).reshape(n, d))
      acc :+ stepForward(v, acc.lastOption.getOrElse(h0), wX, wH, b)
  }

  override def forward(): Variable = {
    val data = ns.zeros(n, t, h)
    for (i <- 0 until t) {
      data(:>, i, :>) := vs(i).data
    }
    Variable(data, Some(this))
  }

  override def backward(gradOutput: Variable): Unit = {
    // val gData = ns.zerosLike(gradOutput.data)
    vs.zipWithIndex.reverse.foreach {
      case (v, i) =>
        val gi = Variable(gradOutput.data(:>, i, :>).reshape(v.shape: _*))
        v.backward(gi)
        // gData(:>, i, :>) := v.grad.get.data.reshape(2, 1, 5)
    }
    // still need to update x.grad!!!!!!!!!
    // x.g = Some(gData)

  }
}

object RnnFunction {
  def stepForward(x: Variable,
                  prevH: Variable,
                  wX: Variable,
                  wH: Variable,
                  b: Variable): Variable = {
    val xWx = x dot wX
    val prevHwH = prevH dot wH
    tanh(xWx + prevHwH + b)
  }
}
