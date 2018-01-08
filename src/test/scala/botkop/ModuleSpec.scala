package botkop

import botkop.autograd.Variable
import botkop.nn.{Linear, Module}
import botkop.{numsca => ns}
import org.scalatest.{FlatSpec, Matchers}

class ModuleSpec extends FlatSpec with Matchers {

  "A Module" should "compute a simple linear network" in {

    case class Net() extends Module {
      val fc = Linear(36, 10)
      override def forward(x: Variable): Variable =
        nn.relu(fc(x))
    }

    val net = Net()

    val input = Variable(ns.randn(36, 8))
    val out = net(input)

    out.data.shape shouldBe Array(10, 8)

    val dout = Variable(ns.randn(out.data.shape))

    out.backward(dout)
    input.grad.get.data.shape shouldBe input.data.shape

    //println(out)

    // println(input.grad)
  }
}
