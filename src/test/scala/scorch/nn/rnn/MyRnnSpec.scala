package scorch.nn.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.Variable
import scorch.nn.Module
import scorch.autograd._
import scorch.nn._

class MyRnnSpec extends FlatSpec with Matchers {

  Nd4j.setDataType(DataBuffer.Type.DOUBLE)
  ns.rand.setSeed(231)

  "My Rnn" should "well dunno..." in {

    case class MyRnnStep(at0: Variable,
                         waa: Variable,
                         wax: Variable,
                         ba: Variable,
                         wya: Variable,
                         by: Variable)
        extends Module {

      def forward(xt: Variable): Variable = {
        val at = tanh(waa.dot(at0) + wax.dot(xt) + ba)
        val yt = sigmoid(wya.dot(at) + by)
        yt
      }
    }

  }

}
