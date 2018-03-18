package scorch.nn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil.oneOpGradientCheck
import scorch.autograd.Variable
import scorch.nn.BatchNorm.{BatchNormFunction, ChainRuleBatchNormFunction}
import scorch.TestUtil._

class BatchNormSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "BatchNorm" should "forward pass with gamma = 1 and beta = 0" in {

    val (n, d1, d2, d3) = (200, 50, 60, 3)
    val x = ns.randn(n, d1)
    val w1 = ns.randn(d1, d2)
    val w2 = ns.randn(d2, d3)
    val a = ns.maximum(x.dot(w1), 0).dot(w2)

    println("Before batch normalization:")
    println(s"  means: ${ns.mean(a, axis = 0)}")
    println(s"  stds: ${ns.std(a, axis = 0)}")

    val bn = BatchNorm(Variable(ns.ones(d3)), Variable(ns.zeros(d3)))
    bn.train()

    val aNorm = bn.forward(Variable(a))
    val meanNorm = ns.mean(aNorm.data, axis = 0)
    val stdNorm = ns.std(aNorm.data, axis = 0)
    println("After batch normalization:  (gamma=1, beta=0)")
    println(s"  means: $meanNorm")
    println(s"  stds: $stdNorm")

    // Means should be close to zero and stds close to one
    val meanError = relError(meanNorm, ns.zeros(d3))
    println(meanError)
    meanError should be < 1e-5
    val stdError = relError(stdNorm, ns.ones(d3))

    println(stdNorm)
    println(stdError)
    stdError should be < 1e-5

  }

  it should "forward pass with non trivial gamma and beta" in {

    val (n, d1, d2, d3) = (200, 50, 60, 3)
    val x = ns.randn(n, d1)
    val w1 = ns.randn(d1, d2)
    val w2 = ns.randn(d2, d3)
    val a = ns.maximum(x.dot(w1), 0).dot(w2)

    println("Before batch normalization:")
    println(s"  means: ${ns.mean(a, axis = 0)}")
    println(s"  stds: ${ns.std(a, axis = 0)}")

    // Now means should be close to beta and stds close to gamma
    val gt = Tensor(1.0, 2.0, 3.0)
    val gamma = Variable(gt)
    val bt = Tensor(11.0, 12.0, 13.0)
    val beta = Variable(bt)

    val bn = BatchNorm(gamma, beta)
    bn.train()

    val aNorm = bn.forward(Variable(a))
    val meanNorm = ns.mean(aNorm.data, axis = 0)
    val stdNorm = ns.std(aNorm.data, axis = 0)
    println("After batch normalization: (nontrivial gamma, beta)")
    println(s"  means: $meanNorm")
    println(s"  stds: $stdNorm")

    // Means should be close to beta and stds close to gamma
    val meanError = relError(meanNorm, bt)
    println(meanError)
    meanError should be < 1e-5

    val stdError = relError(stdNorm, gt)
    println(stdError)
    stdError should be < 1e-5

  }

  it should "train many times, then test" in {

    //Check the test-time forward pass by running the training-time
    //forward pass many times to warm up the running averages, and then
    //checking the means and variances of activations after a test-time
    //forward pass.

    val (n, d1, d2, d3) = (200, 50, 60, 3)
    val w1 = ns.randn(d1, d2)
    val w2 = ns.randn(d2, d3)

    val gt = ns.ones(d3)
    val gamma = Variable(gt)
    val bt = ns.zeros(d3)
    val beta = Variable(bt)

    val bn = BatchNorm(gamma, beta)
    bn.train()

    for (_ <- 1 to 50) {
      val x = ns.randn(n, d1)
      val a = ns.maximum(x.dot(w1), 0).dot(w2)
      bn.forward(Variable(a))
    }

    bn.train(false)
    val x = ns.randn(n, d1)
    val a = ns.maximum(x.dot(w1), 0).dot(w2)
    val aNorm = bn.forward(Variable(a))

    // Means should be close to zero and stds close to one, but will be
    // noisier than training-time forward passes.
    val meanNorm = ns.mean(aNorm.data, axis = 0)
    val stdNorm = ns.std(aNorm.data, axis = 0)
    println("After batch normalization:  (test-time)")
    println(s"  means: $meanNorm")
    println(s"  stds: $stdNorm")

    meanNorm.data.foreach(d => d should be(0.0 +- 0.1))
    stdNorm.data.foreach(d => d should be(1.0 +- 0.2))
  }

  it should "calculate gradients" in {

    val (n, d) = (4, 5)
    val x = Variable(5 * ns.randn(n, d) + 12)
    val gamma = Variable(ns.randn(1, d))
    val beta = Variable(ns.randn(1, d))
    val runningMean: Tensor = ns.zerosLike(gamma.data)
    val runningVar: Tensor = ns.zerosLike(gamma.data)

    def fx(a: Variable): Variable = {
      BatchNormFunction(a,
                        1e-5,
                        0.9,
                        runningMean,
                        runningVar,
                        gamma,
                        beta,
                        inTrainingMode = true).forward()
    }

    def fg(a: Variable): Variable = {
      BatchNormFunction(x,
                        1e-5,
                        0.9,
                        runningMean,
                        runningVar,
                        a,
                        beta,
                        inTrainingMode = true).forward()
    }

    def fb(a: Variable): Variable = {
      BatchNormFunction(x,
                        1e-5,
                        0.9,
                        runningMean,
                        runningVar,
                        gamma,
                        a,
                        inTrainingMode = true).forward()
    }

    oneOpGradientCheck(fx, x)
    oneOpGradientCheck(fg, gamma.copy())
    oneOpGradientCheck(fb, beta.copy())
  }

  it should "calculate gradients using the chain rule" in {

    val (n, d) = (4, 5)
    val x = Variable(5 * ns.randn(n, d) + 12)
    val gamma = Variable(ns.randn(1, d))
    val beta = Variable(ns.randn(1, d))
    val runningMean: Tensor = ns.zerosLike(gamma.data)
    val runningVar: Tensor = ns.zerosLike(gamma.data)

    def fx(a: Variable): Variable = {
      ChainRuleBatchNormFunction(a,
                                 1e-5,
                                 0.9,
                                 runningMean,
                                 runningVar,
                                 gamma,
                                 beta,
                                 inTrainingMode = true).forward()
    }

    def fg(a: Variable): Variable = {
      ChainRuleBatchNormFunction(x,
                                 1e-5,
                                 0.9,
                                 runningMean,
                                 runningVar,
                                 a,
                                 beta,
                                 inTrainingMode = true).forward()
    }

    def fb(a: Variable): Variable = {
      ChainRuleBatchNormFunction(x,
                                 1e-5,
                                 0.9,
                                 runningMean,
                                 runningVar,
                                 gamma,
                                 a,
                                 inTrainingMode = true).forward()
    }

    oneOpGradientCheck(fx, x)
    oneOpGradientCheck(fg, gamma.copy())
    oneOpGradientCheck(fb, beta.copy())
  }

}
