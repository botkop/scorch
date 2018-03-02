package scorch.sandbox.rnn.word

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.nn.rnn.RnnCellBase

import scala.annotation.tailrec

/**
  * Sample a sequence of  according to a sequence of probability distributions output of the RNN
  * @param rnn the network module
  * @param tokenToIx maps tokens to indices
  * @param eolIndex index of the end-of-sentence token
  * @param maxStringSize maximum length of a generated string
  */
case class WordSampler(rnn: RnnCellBase,
                       tokenToIx: Map[String, Int],
                       eolIndex: Int,
                       maxStringSize: Int) {
  val vocabSize: Int = tokenToIx.size
  val ixToToken: Map[Int, String] = tokenToIx.map(_.swap)

  /**
    * Reuse the previously generated token and hidden state to generate the next token
    * @param xPrev the previous token
    * @param pPrev the previous state
    * @return tuple of (next token, index of the next token, the next state)
    */
  def generateNextToken(xPrev: Variable,
                        pPrev: Seq[Variable]): (Variable, Int, Seq[Variable]) = {
    // Forward propagate x
    val next = rnn(xPrev +: pPrev: _*)
    val (yHat, pNext) = (next.head, next.tail)
    // Sample the index of a token within the vocabulary from the probability distribution y
    val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
    // one hot encoding of the next index
    val xNext = Variable(ns.zerosLike(xPrev.data))
    xNext.data(nextIdx, 0) := 1
    (xNext, nextIdx, pNext)
  }

  /**
    * Recurse over time-steps t. At each time-step, sample a token from a probability distribution and append
    * its index to "indices". We'll stop if we reach maxStringSize tokens (which should be very unlikely with a well
    * trained model), which helps debugging and prevents entering an infinite loop.
    * @param t current time step
    * @param prevX previous token
    * @param prev previous activation
    * @param indices accumulator of indices
    * @return indices
    */
  @tailrec
  final def generate(t: Int,
                     prevX: Variable,
                     prev: Seq[Variable],
                     indices: List[Int] = List.empty): List[Int] =
    if (indices.lastOption.contains(eolIndex)) {
      indices
    } else if (t >= maxStringSize) {
      indices :+ eolIndex
    } else {
      val (nextX, nextIdx, nextP) = generateNextToken(prevX, prev)
      generate(t + 1, nextX, nextP, indices :+ nextIdx)
    }

  def sample(join: (Seq[String]) => String): String = {
    val x0 = Variable(ns.zeros(vocabSize, 1))
    val p0 = rnn.initialTrackingStates
    val sampledIndices = generate(1, x0, p0)
    join(sampledIndices.init.map(ixToToken))
  }
}
