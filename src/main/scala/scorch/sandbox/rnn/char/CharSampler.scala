package scorch.sandbox.rnn.char

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.nn.rnn.RnnCellBase

import scala.annotation.tailrec

/**
  * Sample a sequence of characters according to a sequence of probability distributions output of the RNN
  * @param rnnCell the network module
  * @param charToIx maps characters to indices
  * @param eolIndex index of the end-of-line character
  * @param maxStringSize maximum length of a generated string
  */
case class CharSampler(rnnCell: RnnCellBase,
                       charToIx: Map[Char, Int],
                       eolIndex: Int,
                       maxStringSize: Int) {
  val vocabSize: Int = charToIx.size
  val ixToChar: Map[Int, Char] = charToIx.map(_.swap)

  /**
    * Reuse the previously generated character and hidden state to generate the next character
    * @param xPrev the previous character
    * @param pPrev the previous state
    * @return tuple of (next character, index of the next character, the next state)
    */
  def generateNextChar(xPrev: Variable,
                       pPrev: Seq[Variable]): (Variable, Int, Seq[Variable]) = {
    // Forward propagate X
    val next = rnnCell(xPrev +: pPrev: _*)
    val (yHat, pNext) = (next.head, next.tail)
    // Sample the index of a character within the vocabulary from the probability distribution y
    val nextIdx = ns.choice(ns.arange(vocabSize), yHat.data).squeeze().toInt
    // one hot encoding of the next index
    val xNext = Variable(ns.zerosLike(xPrev.data))
    xNext.data(nextIdx, 0) := 1
    (xNext, nextIdx, pNext)
  }

  /**
    * Recurse over time-steps t. At each time-step, sample a character from a probability distribution and append
    * its index to "indices". We'll stop if we reach maxStringSize characters (which should be very unlikely with a well
    * trained model), which helps debugging and prevents entering an infinite loop.
    * @param t current time step
    * @param prevX previous character
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
      val (nextX, nextIdx, nextP) = generateNextChar(prevX, prev)
      generate(t + 1, nextX, nextP, indices :+ nextIdx)
    }

  def sample: String = {
    val x0 = Variable(ns.zeros(vocabSize, 1))
    val p0 = rnnCell.initialTrackingStates
    val sampledIndices = generate(1, x0, p0)
    sampledIndices.map(ixToChar).mkString
  }
}
