package scorch.nn.rnn

import botkop.{numsca => ns}
import scorch.autograd.{Variable, sigmoid, softmax, tanh}

/**
  * Implements a single forward step of the LSTM-cell
  * @param wf Weight matrix of the forget gate, numpy array of shape (na, na + nx)
  * @param bf Bias of the forget gate, numpy array of shape (na, 1)
  * @param wi Weight matrix of the update gate, numpy array of shape (na, na + nx)
  * @param bi Bias of the update gate, numpy array of shape (na, 1)
  * @param wc Weight matrix of the first "tanh", numpy array of shape (na, na + nx)
  * @param bc Bias of the first "tanh", numpy array of shape (na, 1)
  * @param wo Weight matrix of the output gate, numpy array of shape (na, na + nx)
  * @param bo Bias of the output gate, numpy array of shape (na, 1)
  * @param wy Weight matrix relating the hidden-state to the output, numpy array of shape (ny, na)
  * @param by Bias relating the hidden-state to the output, numpy array of shape (ny, 1)
  */
case class LstmCell(
    wf: Variable,
    bf: Variable,
    wi: Variable,
    bi: Variable,
    wc: Variable,
    bc: Variable,
    wo: Variable,
    bo: Variable,
    wy: Variable,
    by: Variable
) extends BaseRnnCell(Seq(wf, bf, wi, bi, wc, bc, wo, bo, wy, by)) {
  override val na: Int = wy.shape.last
  override val numTrackingStates: Int = 2

  /**
    * Lstm cell forward pass
    * @param xs sequence of Variables:
    *           - xt: your input data at timestep "t", of shape (nx, m).
    *           - aPrev: Hidden state at timestep "t-1", of shape (na, m)
    *           - cPrev: Memory state at timestep "t-1", numpy array of shape (na, m)
    * @return sequence of Variables:
    *         - aNext: next hidden state, of shape (na, m)
    *         - cNext: next memory state, of shape (na, m)
    *         - ytHat: prediction at timestep "t", of shape (ny, m)
    */
  override def forward(xs: Seq[Variable]): Seq[Variable] = xs match {
    case Seq(xt, aPrev, cPrev) =>
      val concat = scorch.cat(aPrev, xt)

      // Forget gate
      val ft = sigmoid(wf.dot(concat) + bf)
      // Update gate
      val it = sigmoid(wi.dot(concat) + bi)
      val cct = tanh(wc.dot(concat) + bc)
      val cNext = ft * cPrev + it * cct
      // Output gate
      val ot = sigmoid(wo.dot(concat) + bo)
      val aNext = ot * tanh(cNext)
      val ytHat = softmax(wy.dot(aNext) + by)
      Seq(ytHat, aNext.detach(), cNext.detach())
  }
}

object LstmCell {

  /**
    * Create an LstmCell from dimensions
    * @param na number of units of the LstmCell
    * @param nx size of the weight matrix multiplying the input
    * @param ny size of the weight matrix relating the hidden-state to the output
    * @return initialized Lstm cell
    */
  def apply(na: Int, nx: Int, ny: Int): LstmCell = {

    val i = math.sqrt(2.0 / na)

    val wf = Variable(ns.randn(na, na + nx) * i, name = Some("wf"))
    val bf = Variable(ns.zeros(na, 1), name = Some("bf"))
    val wi = Variable(ns.randn(na, na + nx) * i, name = Some("wi"))
    val bi = Variable(ns.zeros(na, 1), name = Some("bi"))
    val wc = Variable(ns.randn(na, na + nx) * i, name = Some("wc"))
    val bc = Variable(ns.zeros(na, 1), name = Some("bc"))
    val wo = Variable(ns.randn(na, na + nx) * i, name = Some("bo"))
    val bo = Variable(ns.zeros(na, 1), name = Some("by"))
    val wy = Variable(ns.randn(ny, na) * math.sqrt(2.0 / ny), name = Some("wy"))
    val by = Variable(ns.zeros(ny, 1), name = Some("ba"))
    LstmCell(wf, bf, wi, bi, wc, bc, wo, bo, wy, by)
  }
}
