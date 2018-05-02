package scorch.nn.cnn

import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.Variable
import scorch.TestUtil.oneOpGradientCheck

class MaxPool2dSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(234)

  "NaiveMaxPoolingFunction" should "pass forward" in {
    val xShape = List(2, 3, 4, 4)
    val x = Variable(ns.linspace(-0.3, 0.4, xShape.product).reshape(xShape: _*))
    val poolWidth = 2
    val poolHeight = 2
    val stride = 2

    val out = MaxPool2d
      .NaiveMaxPool2dFunction(x, poolHeight, poolWidth, stride)
      .forward()

    val correctOut = ns
      .array(-0.26315789, -0.24842105, -0.20421053, -0.18947368, -0.14526316,
        -0.13052632, -0.08631579, -0.07157895, -0.02736842, -0.01263158,
        0.03157895, 0.04631579, 0.09052632, 0.10526316, 0.14947368, 0.16421053,
        0.20842105, 0.22315789, 0.26736842, 0.28210526, 0.32631579, 0.34105263,
        0.38526316, 0.4)
      .reshape(2, 3, 2, 2)

    val error = scorch.TestUtil.relError(out.data, correctOut)
    error should be < 5e-8
  }

  it should "pass backward" in {
    val x = Variable(ns.randn(3, 2, 8, 8))
    val poolWidth = 2
    val poolHeight = 2
    val stride = 2

    def fx(a: Variable): Variable =
      MaxPool2d
        .NaiveMaxPool2dFunction(a, poolHeight, poolWidth, stride)
        .forward()

    oneOpGradientCheck(fx, x)
  }


  "ReshapeMaxPool2dFunction" should "pass forward" in {
    val xShape = List(2, 3, 4, 4)
    val x = Variable(ns.linspace(-0.3, 0.4, xShape.product).reshape(xShape: _*))
    val poolWidth = 2
    val poolHeight = 2
    val stride = 2

    val out = MaxPool2d
      .ReshapeMaxPool2dFunction(x, poolHeight, poolWidth, stride)
      .forward()

    val correctOut = ns
      .array(-0.26315789, -0.24842105, -0.20421053, -0.18947368, -0.14526316,
        -0.13052632, -0.08631579, -0.07157895, -0.02736842, -0.01263158,
        0.03157895, 0.04631579, 0.09052632, 0.10526316, 0.14947368, 0.16421053,
        0.20842105, 0.22315789, 0.26736842, 0.28210526, 0.32631579, 0.34105263,
        0.38526316, 0.4)
      .reshape(2, 3, 2, 2)

    val error = scorch.TestUtil.relError(out.data, correctOut)
    error should be < 5e-8
  }

  it should "pass backward" in {
    val x = Variable(ns.randn(3, 2, 8, 8))
    val poolWidth = 2
    val poolHeight = 2
    val stride = 2

    def fx(a: Variable): Variable =
      MaxPool2d
        .ReshapeMaxPool2dFunction(a, poolHeight, poolWidth, stride)
        .forward()

    oneOpGradientCheck(fx, x)
  }

}
