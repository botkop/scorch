package scorch.nn
import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}

case class BatchNorm(gamma: Variable,
                     beta: Variable,
                     eps: Double = 1e-5,
                     momentum: Double = 0.9)
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
  def apply(d: Int, eps: Double, momentum: Double): BatchNorm = {
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

    import scorch._

    override def forward(): Variable = {

      if (inTrainingMode) {
        val mu = mean(x, axis = 0)
        val v = variance(x, axis = 0)

        runningMean := (momentum * runningMean) + ((1.0 - momentum) * mu.data)

        println(runningVar.shape.toList)
        println(x.shape)

        runningVar := (momentum * runningVar) + ((1.0 - momentum) * v.data)


        val xMu = x - mu
        val invVar = 1.0 / sqrt(v + eps)
        val xHat = xMu * invVar
        (gamma * xHat) + beta

      }
      else {
        val out = ((x.data - runningMean) / ns.sqrt(runningVar + eps)) * gamma.data + beta.data
        Variable(out, gradFn = Some(this))
      }
    }

    override def backward(gradOutput: Variable): Unit = {
      x.backward(gradOutput)
    }
  }

  /*
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

    // all below variables are needed in training phase only
    // making them lazy, so they don't get evaluated in test phase

    // compute per-dimension mean and std deviation
    val mean: Tensor = ns.mean(x.data, axis = 0)
    val variance: Tensor = ns.variance(x.data, axis = 0)

    // normalize and zero-center (explicit for caching purposes)
    val xMu: Tensor = x.data - mean
    val invVar: Tensor = 1.0 / ns.sqrt(variance + eps)

    val xHat: Tensor = xMu * invVar

    override def forward(): Variable =
      if (inTrainingMode) {
        runningMean := (momentum * runningMean) + ((1.0 - momentum) * mean)
        runningVar := (momentum * runningVar) + ((1.0 - momentum) * variance)

        // squash
        val out = (xHat * gamma.data) + beta.data
        Variable(out, gradFn = Some(this))
      } else {
        val out = ((x.data - runningMean) / ns.sqrt(runningVar + eps)) * gamma.data + beta.data
        Variable(out, gradFn = Some(this))
      }

    override def backward(gradOutput: Variable): Unit = {
      val dOut = gradOutput.data

      beta.grad.data := ns.sum(dOut, axis = 0)
      gamma.grad.data := ns.sum(xHat * dOut, axis = 0)

      // intermediate partial derivatives
      val dxHat = dOut * gamma.data
      val dx = (invVar / n) * ((dxHat * n) - ns.sum(dxHat, axis = 0) -
        (xHat * ns.sum(dxHat * xHat, axis = 0)))

      x.backward(Variable(dx))
    }
  }
  */

}
