package scorch.nn.rnn

import botkop.{numsca => ns}
import scorch.autograd.{Variable, softmax, tanh}

/**
  * Module with vanilla RNN activation.
  * @param wax Weight matrix multiplying the input, variable of shape (na, nx)
  * @param waa Weight matrix multiplying the hidden state, variable of shape (na, na)
  * @param wya Weight matrix relating the hidden-state to the output, variable of shape (ny, na)
  * @param ba Bias, of shape (na, 1)
  * @param by Bias relating the hidden-state to the output, of shape (ny, 1)
  */
case class RnnCell(wax: Variable,
                   waa: Variable,
                   wya: Variable,
                   ba: Variable,
                   by: Variable)
    extends BaseRnnCell(Seq(wax, waa, wya, ba, by)) {

  override val na: Int = wax.shape.head
  override val numTrackingStates: Int = 1

  override def forward(xs: Seq[Variable]): Seq[Variable] = xs match {
    case Seq(xt, aPrev) =>
      val aNext = tanh(waa.dot(aPrev) + wax.dot(xt) + ba)
      val yt = softmax(wya.dot(aNext) + by)
      Seq(yt, aNext)
  }
}

object RnnCell {
  /**
    * Create an RnnCell from dimensions
    * @param na number of units of the RNN cell
    * @param nx size of the weight matrix multiplying the input
    * @param ny size of the weight matrix relating the hidden-state to the output
    * @return a vanilla Rnn model
    */
  def apply(na: Int, nx: Int, ny: Int): RnnCell = {
    val wax = Variable(ns.randn(na, nx) * 0.01, name = Some("wax"))
    val waa = Variable(ns.randn(na, na) * 0.01, name = Some("waa"))
    val wya = Variable(ns.randn(ny, na) * 0.01, name = Some("wya"))
    val ba = Variable(ns.zeros(na, 1), name = Some("ba"))
    val by = Variable(ns.zeros(ny, 1), name = Some("by"))
    RnnCell(wax, waa, wya, ba, by)
  }
}
