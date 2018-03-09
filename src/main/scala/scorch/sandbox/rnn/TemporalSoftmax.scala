package scorch.sandbox.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd._
import scorch.nn.SimpleModule

case class TemporalSoftmax(y: Variable, mask: Variable) extends SimpleModule {
  override def forward(x: Variable): Variable =
    TemporalSoftmaxFunction(x, y, mask).forward()
}

/**
A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
  */
case class TemporalSoftmaxFunction(x: Variable, y: Variable, mask: Variable)
    extends Function {

  val List(n, t, v) = x.shape
  val xFlat: Tensor = x.data.reshape(n * t, v)
  val yFlat: Tensor = y.data.reshape(1, n * t)
  val maskFlat: Tensor = mask.data.reshape(1, n * t)

  val probs: Tensor = ns.exp(xFlat - ns.max(xFlat, axis = 1))
  probs /= ns.sum(probs, axis = 1)

  val loss: Double =
    -ns.sum(maskFlat * ns.log(probs(ns.arange(n * t), yFlat))) / n

  val dxFlat: Tensor = probs
  dxFlat(ns.arange(n * t), yFlat) -= 1
  dxFlat /= n
  dxFlat *= maskFlat.T

  val dx: Tensor = dxFlat.reshape(n, t, v)

  override def forward(): Variable = Variable(Tensor(loss), Some(this))

  override def backward(gradOutput: Variable /* not used */ ): Unit = {
    x.backward(Variable(dx))
  }
}
