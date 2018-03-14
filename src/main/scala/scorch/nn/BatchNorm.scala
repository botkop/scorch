package scorch.nn
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}

case class BatchNorm(shape: List[Int],
                     eps: Double = 1e-5,
                     momentum: Double = 0.9)
    extends Module {

  val List(d, n) = shape

  val runningMean: Tensor = ns.zeros(d, 1)
  val runningVar: Tensor = ns.zeros(d, 1)
  val gamma: Tensor = ns.ones(d, 1)
  val beta: Tensor = ns.zeros(d, 1)

  override def forward(x: Variable): Variable =
    BatchNormFunction(x,
                      n,
                      eps,
                      momentum,
                      runningMean,
                      runningVar,
                      gamma,
                      beta,
                      inTrainingMode)
      .forward()

}

case class BatchNormFunction(x: Variable,
                             n: Int,
                             eps: Double,
                             momentum: Double,
                             runningMean: Tensor,
                             runningVar: Tensor,
                             gamma: Tensor,
                             beta: Tensor,
                             inTrainingMode: Boolean)
    extends Function {

  var xHat = Tensor(0)
  var invVar = Tensor(0)

  override def forward(): Variable =
    if (inTrainingMode) {
      // compute per-dimension mean and std_deviation
      val mean = ns.mean(x.data, axis = 1)
      val variance = ns.variance(x.data, axis = 1)

      // normalize and zero-center (explicit for caching purposes)
      val xMu = x.data - mean
      invVar = 1.0 / ns.sqrt(variance + eps)
      xHat = xMu * invVar

      // squash
      val out = xHat * gamma + beta

      runningMean *= momentum
      runningMean += (1 - momentum) * mean
      runningVar *= momentum
      runningVar += (1 - momentum) * variance

      Variable(out, gradFn = Some(this))
    } else {
      val out = ((x.data - runningMean) / ns.sqrt(runningVar + eps)) * gamma + beta
      Variable(out, gradFn = Some(this))
    }

  override def backward(gradOutput: Variable): Unit =
    if (inTrainingMode) {
      val dOut = gradOutput.data

      // intermediate partial derivatives
      val dxHat = dOut * gamma

      // final partial derivatives
      val dx = ((n * dxHat) - ns.sum(dxHat, axis = 1) - (xHat * ns
        .sum(dxHat * xHat, axis = 1))) * invVar * (1.0 / n)

      val dBeta = ns.sum(dOut, axis = 1)
      val dGamma = ns.sum(xHat * dOut, axis = 1)

      // not sure about this...
      beta -= dBeta
      gamma -= dGamma

      x.backward(Variable(dx))
    } else {
      x.backward(gradOutput)
    }
}
