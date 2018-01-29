package scorch

import botkop.{numsca => ns}
import botkop.numsca.Tensor

object TestUtil {

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

}
