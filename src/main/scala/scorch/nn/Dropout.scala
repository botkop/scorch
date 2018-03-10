package scorch.nn

import scorch.autograd.{DropoutFunction, Variable}

case class Dropout(p: Double = 0.5) extends Module {
  override def forward(x: Variable): Variable =
    DropoutFunction(x, p, inTrainingMode).forward()
}
