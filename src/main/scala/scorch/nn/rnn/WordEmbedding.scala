package scorch.nn.rnn

import botkop.numsca.Tensor
import botkop.{numsca => ns}
import scorch.autograd._
import scorch.nn.Module

import scala.language.postfixOps

case class WordEmbedding(w: Variable) extends Module(Seq(w)) {
  override def forward(x: Variable): Variable =
    WordEmbeddingFunction(x, w).forward()
}

object WordEmbedding {
  def apply(v: Int, d: Int): WordEmbedding = {
    val w = ns.randn(v, d)
    WordEmbedding(Variable(w))
  }
}

case class WordEmbeddingFunction(x: Variable, w: Variable) extends Function {

  val List(n, t) = x.shape
  val List(v, d) = w.shape

  override def forward(): Variable = {
    // todo: write idiomatic implementation in numsca
    val out = for {
      i <- 0 until n
      j <- 0 until t
    } yield {
      val wix = x.data(i, j).squeeze().toInt
      w.data(wix).data
    }

    val td = Tensor(out.toArray.flatten).reshape(n, t, d)

    Variable(td, Some(this))
  }

  override def backward(gradOutput: Variable): Unit = {

    // x = n * t
    // w = v * d
    // g = n * t * d

    val dW = ns.zerosLike(w.data)

    for {
      i <- 0 until n
      j <- 0 until t
    } {
      val gWord = gradOutput.data(i, j).reshape(1, d)
      val wordIx = x.data(i, j).squeeze().toInt
      dW(wordIx) += gWord
    }

    w.backward(Variable(dW))
    x.backward()
  }

}
