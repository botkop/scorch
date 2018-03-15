package scorch.nn
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}

case class BatchNorm(gamma: Variable,
                     beta: Variable,
                     eps: Double,
                     momentum: Double)
    extends Module(Seq(gamma, beta)) {

  import BatchNorm._

  val runningMean: Tensor = ns.zerosLike(gamma.data)
  val runningVar: Tensor = ns.zerosLike(gamma.data)

  override def forward(x: Variable): Variable =
    BatchNormFunction(x,
                      eps,
                      momentum,
                      runningMean,
                      runningVar,
                      gamma,
                      beta,
                      inTrainingMode)
      .forward()
}

object BatchNorm {
  def apply(shape: List[Int],
            eps: Double = 1e-5,
            momentum: Double = 0.9): BatchNorm = {
    val d = shape.head
    val gamma = Variable(ns.ones(1, d))
    val beta = Variable(ns.zeros(1, d))
    BatchNorm(gamma, beta, eps, momentum)
  }

  case class BatchNormFunction(x: Variable,
                               eps: Double,
                               momentum: Double,
                               runningMean: Tensor,
                               runningVar: Tensor,
                               gamma: Variable,
                               beta: Variable,
                               inTrainingMode: Boolean)
      extends Function {

    val List(n, d) = x.shape

    // all below variables are only needed in training phase
    // making them lazy, so they don't get evaluated in test phase

    // compute per-dimension mean and std deviation
    lazy val mean: Tensor = ns.mean(x.data, axis = 0)
    lazy val variance: Tensor = ns.variance(x.data, axis = 0)

    // normalize and zero-center (explicit for caching purposes)
    lazy val xMu: Tensor = x.data - mean
    lazy val invVar: Tensor = 1.0 / ns.sqrt(variance + eps)
    lazy val xHat: Tensor = xMu * invVar

    override def forward(): Variable =
      if (inTrainingMode) {
        // squash
        val out: Tensor = xHat * gamma.data + beta.data

        runningMean *= momentum

        runningMean += (1 - momentum) * mean
        runningVar *= momentum
        runningVar += (1 - momentum) * variance

        Variable(out, gradFn = Some(this))
      } else {
        val out = ((x.data - runningMean) / ns.sqrt(runningVar + eps)) * gamma.data + beta.data
        Variable(out, gradFn = Some(this))
      }

    override def backward(gradOutput: Variable): Unit = {
      val dOut = gradOutput.data

      // intermediate partial derivatives
      val dxHat = dOut * gamma.data

      // final partial derivatives
      val dx = ((n * dxHat) - ns.sum(dxHat, axis = 0) -
        (xHat * ns.sum(dxHat * xHat, axis = 0))) * invVar * (1.0 / n)

      beta.grad.data := ns.sum(dOut, axis = 0)
      gamma.grad.data := ns.sum(xHat * dOut, axis = 0)

      x.backward(Variable(dx))
    }
  }
}
