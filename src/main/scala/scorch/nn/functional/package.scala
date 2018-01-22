package scorch.nn

import scorch.autograd.{Dropout, Threshold, Variable}

package object functional {

  def threshold(v: Variable, d: Double): Variable = Threshold(v, d).forward()

  def dropout(v: Variable, p: Double = 0.5, train: Boolean): Variable =
    Dropout(v, p, train).forward()
}
