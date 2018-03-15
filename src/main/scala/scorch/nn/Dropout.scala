package scorch.nn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd.{Function, Variable}
import scorch.nn.Dropout.DropoutFunction

case class Dropout(p: Double = 0.5) extends Module {
  override def forward(x: Variable): Variable =
    DropoutFunction(x, p, inTrainingMode).forward()
}

object Dropout {

  case class DropoutFunction(x: Variable,
                             p: Double = 0.5,
                             train: Boolean = false,
                             maybeMask: Option[Tensor] = None)
      extends Function {

    require(p > 0 && p < 1,
            s"dropout probability has to be between 0 and 1, but got $p")

    // maybeMask can be provided for testing purposes
    val mask: Tensor = maybeMask.getOrElse {
      (ns.rand(x.shape: _*) < p) / p
    }

    override def forward(): Variable =
      if (train)
        Variable(x.data * mask, Some(this))
      else
        Variable(x.data, Some(this))

    override def backward(gradOutput: Variable): Unit =
      if (train)
        x.backward(Variable(gradOutput.data * mask))
      else
        x.backward(gradOutput)
  }
}
