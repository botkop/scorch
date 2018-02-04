package scorch.nn.rnn

import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil._
import scorch.autograd._

import scala.language.postfixOps

class WordEmbeddingSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "WordEmbedding" should "forward pass" in {
    val (n, t, v, d) = (2, 4, 5, 3)

    val x = Variable(
      ns.array(0, 3, 1, 2, //
          2, 1, 0, 3)
        .reshape(n, t))

    val w = Variable(ns.linspace(0, 1, num = v * d).reshape(v, d))

    val out = WordEmbeddingFunction(x, w).forward()

    val expectedOut = ns
      .array( //
          0, 0.07142857, 0.14285714, //
          0.64285714, 0.71428571, 0.78571429, //
          0.21428571, 0.28571429, 0.35714286, //
          0.42857143, 0.5, 0.57142857, //
          0.42857143, 0.5, 0.57142857, //
          0.21428571, 0.28571429, 0.35714286, //
          0, 0.07142857, 0.14285714, //
          0.64285714, 0.71428571, 0.78571429)
      .reshape(n, t, d)

    val error = relError(out.data, expectedOut)
    println(error)
    assert(error < 1e-7)
  }

  it should "backward pass" in {
    val (n, t, v, d) = (50, 3, 5, 6)
    val x = Variable(ns.randint(v, Array(n, t)))
    val w = Variable(ns.randn(v, d))

    def f(a: Variable): Variable = WordEmbeddingFunction(x, a).forward()
    oneOpGradientCheck(f, w)
  }

}


