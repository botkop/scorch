package scorch

import botkop.{numsca => ns}
import botkop.numsca.Tensor
import com.typesafe.scalalogging.LazyLogging
import scorch.autograd.Variable

object TestUtil extends LazyLogging {

  def relError(x: Tensor, y: Tensor): Double =
    ns.max(ns.abs(x - y) / ns.maximum(ns.abs(x) + ns.abs(y), 1e-8)).squeeze()

  def evalNumericalGradientArray(f: (Tensor) => Tensor,
                                 x: Tensor,
                                 df: Tensor,
                                 h: Double = 1e-5): Tensor = {
    val grad = ns.zeros(x.shape)

    val it = ns.nditer(x)
    while (it.hasNext) {
      val ix = it.next

      val oldVal = x(ix).squeeze()

      x(ix) := oldVal + h
      val pos = f(x)
      x(ix) := oldVal - h
      val neg = f(x)
      x(ix) := oldVal

      val g = ns.sum((pos - neg) * df) / (2.0 * h)

      grad(ix) := g
    }
    grad
  }

  def binOpGradientCheck(f: (Variable, Variable) => Variable,
                         a: Variable,
                         b: Variable,
                         dOutOpt: Option[Variable] = None): Unit = {

    val out = f(a, b)
    logger.debug(s"out = $out")

    val dOut = dOutOpt.getOrElse(Variable(ns.abs(ns.randn(out.shape: _*))))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)

    val da = a.grad.get.data
    val db = b.grad.get.data

    logger.debug(s"da = $da")
    logger.debug(s"db = $db")

    def fa(t: Tensor) = f(Variable(t), b).data
    def fb(t: Tensor) = f(a, Variable(t)).data

    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    val dbNum = evalNumericalGradientArray(fb, b.data, dOut.data)

    logger.debug(s"daNum = $daNum")
    logger.debug(s"dbNum = $dbNum")

    val daError = relError(da, daNum)
    val dbError = relError(db, dbNum)

    logger.debug(s"daError = $daError")
    logger.debug(s"dbError = $dbError")

    assert(daError < 1e-5)
    assert(dbError < 1e-5)
  }

  def varConstOpGradientCheck(f: (Variable, Double) => Variable,
                              a: Variable,
                              b: Double): Unit = {
    val out = f(a, b)
    logger.debug(s"out = $out")

    val dOut = Variable(ns.randn(out.shape: _*))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)
    val da = a.grad.get.data
    logger.debug(s"da = $da")

    def fa(t: Tensor) = f(Variable(t), b).data
    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    logger.debug(s"daNum = $daNum")

    val daError = relError(da, daNum)
    logger.debug(s"daError = $daError")

    assert(daError < 1e-5)
  }

  def oneOpGradientCheck(f: (Variable) => Variable, a: Variable): Unit = {

    val out = f(a)
    logger.debug(s"out = $out")

    val dOut = Variable(ns.randn(out.shape: _*))
    logger.debug(s"dOut = $dOut")

    out.backward(dOut)

    val da = a.grad.get.data
    logger.debug(s"da = $da")

    def fa(t: Tensor) = f(Variable(t)).data

    val daNum = evalNumericalGradientArray(fa, a.data, dOut.data)
    logger.debug(s"daNum = $daNum")

    val daError = relError(da, daNum)
    logger.debug(s"daError = $daError")

    assert(daError < 1e-5)
  }

}
