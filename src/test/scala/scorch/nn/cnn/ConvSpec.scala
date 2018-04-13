package scorch.nn.cnn

import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.TestUtil.oneOpGradientCheck
import scorch.autograd.Variable
import scorch.nn.{Linear, Module}
import scorch.optim.SGD

import scala.language.implicitConversions

class ConvSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "NaiveConvFunction" should "pass forward" in {

    val xShape = List(2, 3, 4, 4)
    val wShape = List(3, 3, 4, 4)
    val x =
      Variable(ns.linspace(-0.1, 0.5, num = xShape.product).reshape(xShape: _*))
    val w =
      Variable(ns.linspace(-0.2, 0.3, num = wShape.product).reshape(wShape: _*))
    val b = Variable(ns.linspace(-0.1, 0.2, num = 3))
    val stride = 2
    val pad = 1

    val out = Conv.NaiveConvFunction(x, w, b, stride, pad).forward()

    val correctOut = ns
      .array(-0.08759809, -0.10987781, -0.18387192, -0.2109216, 0.21027089,
        0.21661097, 0.22847626, 0.23004637, 0.50813986, 0.54309974, 0.64082444,
        0.67101435, -0.98053589, -1.03143541, -1.19128892, -1.24695841,
        0.69108355, 0.66880383, 0.59480972, 0.56776003, 2.36270298, 2.36904306,
        2.38090835, 2.38247847)
      .reshape(2, 3, 2, 2)

    val error = scorch.TestUtil.relError(out.data, correctOut)
    error should be < 3e-8
  }

  it should "pass backward" in {
    val x = Variable(ns.randn(4, 3, 5, 5))
    val w = Variable(ns.randn(2, 3, 3, 3))
    val b = Variable(ns.randn(2))
    val stride = 1
    val pad = 1

    def fx(a: Variable) = Conv.NaiveConvFunction(a, w, b, stride, pad).forward()
    def fw(a: Variable) = Conv.NaiveConvFunction(x, a, b, stride, pad).forward()
    def fb(a: Variable) = Conv.NaiveConvFunction(x, w, a, stride, pad).forward()

    oneOpGradientCheck(fx, x)
    oneOpGradientCheck(fw, w.copy())
    oneOpGradientCheck(fb, b.copy())
  }

  "A Conv net" should "calculate gradients" in {

    import scorch._

    val weightScale = 1
    val numDataPoints = 2
    val numChannels = 3
    val imageHeight = 8
    val imageWidth = 8
    val numFilters = 3
    val filterSize = 3
    val stride = 1
    val pad = 1

    val poolSize = 2
    val poolStride = 2

    case class ConvReluPool() extends Module {
      val conv = nn.cnn.Conv(numChannels,
                             numFilters,
                             filterSize,
                             weightScale,
                             stride,
                             pad)
      override def subModules = Seq(conv)
      def pool(v: Variable): Variable = maxPool(v, poolSize, poolStride)

      override def forward(x: Variable): Variable = pool(relu(conv(x)))
    }

    val net = ConvReluPool()

    val x =
      Variable(ns.randn(numDataPoints, numChannels, imageHeight, imageWidth))
    def fx(a: Variable) = net(a)
    oneOpGradientCheck(fx, x, 1e-5)
  }

  it should "handle a 3 layer network" in {

    import scorch._

    val weightScale = 1e-3

    // input layer
    val numSamples = 10
    val numChannels = 3
    val imageHeight = 32
    val imageWidth = 32
    val inputShape = List(numSamples, numChannels, imageHeight, imageWidth)

    // convolution layer
    val numFilters = 32
    val filterSize = 7
    val stride = 1
    val pad = 1

    // pooling layer
    val poolSize = 2
    val poolStride = 2

    // hidden layer
    val hiddenDim = 100

    // output layer
    val numClasses = 10

    case class ThreeLayerNetwork() extends Module {

      val conv = nn.cnn.Conv(numChannels,
                             numFilters,
                             filterSize,
                             weightScale,
                             stride,
                             pad)

      val pool = nn.cnn.MaxPooling(poolSize, poolStride)

      val convOutShape: List[Int] = conv.outputShape(inputShape, pad, stride)
      val poolOutShape: List[Int] = pool.outputShape(convOutShape)
      val numFlatFeatures: Int = poolOutShape.tail.product // all dimensions except the batch dimension

      def flatten(v: Variable): Variable =
        v.reshape(numSamples, numFlatFeatures)

      val fc1 = Linear(numFlatFeatures, hiddenDim)
      val fc2 = Linear(hiddenDim, numClasses)

      override def forward(x: Variable): Variable =
        x ~> conv ~> relu ~> pool ~> flatten ~> fc1 ~> fc2
    }

    val net = ThreeLayerNetwork()
    val optimizer = SGD(net.parameters, lr = 0.001)
    val input = Variable(ns.randn(inputShape: _*))
    val target = Variable(ns.randint(numClasses, Array(numSamples, 1)))

    for (j <- 0 to 3) {
      optimizer.zeroGrad()

      val output = net(input)
      val loss = softmaxLoss(output, target)

      val guessed = ns.argmax(output.data, axis = 1)
      val accuracy = ns.sum(target.data == guessed) / numSamples
      println(s"$j: loss: ${loss.data.squeeze()} accuracy: $accuracy")

      loss.backward()
      optimizer.step()
    }
  }
}
