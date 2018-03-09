package scorch.nn.rnn

import scorch.autograd.Variable
import scorch.nn._

import scala.annotation.tailrec

/**
  * Wrapper around an Rnn cell to facilitate froward propagation through a fixed sequence (training)
  * and sampling of results.
  * @param cell the cell base underlying this Rnn
  */
case class RnnBase(cell: RnnCellBase) extends Module[Seq](cell.parameters) {

  /**
    * Performs the forward propagation through the RNN
    * @param xs sequence of variables to activate
    * @return predictions of the RNN over xs
    */
  override def forward(xs: Seq[Variable]): Seq[Variable] =
    xs.foldLeft(List.empty[Variable], cell.initialTrackingStates) {
        case ((yhs, p0), x) =>
          val next = cell(x +: p0)
          val (yht, p1) = (next.head, next.tail)
          (yhs :+ yht, p1)
      }
      ._1

  /**
    * Sample a sequence of symbol indices according to the sequence of probability distributions output of the RNN
    * Terminology: encoded index = symbol
    * @param encode encode an index to a Variable, for example with one-hot encoding
    * @param bosIndex index for the beginning on sentence symbol
    * @param eosIndex index for the end of sentence symbol
    * @param maxSentenceSize maximum length of a sentence
    * @return a sequence of indices terminated by eosIndex and of maximum length maxSentenceSize
    */
  def sample(encode: (Int) => Variable,
             bosIndex: Int,
             eosIndex: Int,
             maxSentenceSize: Int): Seq[Int] = {

    import botkop.{numsca => ns}

    /**
      * Reuse the previously generated symbol and hidden state to generate the next symbol
      * @param xPrev the previous encoded index
      * @param pPrev the previous state
      * @return tuple of (next character, index of the next character, the next state)
      */
    def generateToken(xPrev: Variable,
                      pPrev: Seq[Variable]): (Variable, Int, Seq[Variable]) = {
      // Forward propagate x
      val next = cell(xPrev +: pPrev)
      val (yHat, pNext) = (next.head, next.tail)
      // Sample the index of a token within the vocabulary from the probability distribution y
      val nextIdx =
        ns.choice(ns.arange(yHat.shape.head), yHat.data).squeeze().toInt
      // encoding of the next index
      val xNext = encode(nextIdx)
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
    def generateSequence(t: Int = 1,
                         prevX: Variable = encode(bosIndex),
                         prev: Seq[Variable] = cell.initialTrackingStates,
                         indices: List[Int] = List.empty): List[Int] =
      if (indices.lastOption.contains(eosIndex)) {
        indices
      } else if (t >= maxSentenceSize) {
        indices :+ eosIndex
      } else {
        val (nextX, nextIdx, nextP) = generateToken(prevX, prev)
        generateSequence(t + 1, nextX, nextP, indices :+ nextIdx)
      }

    generateSequence()
  }

}

object RnnBase {
  /**
    * Instantiate an RnnBase object of type defined by rnnType, and given size
    * @param rnnType "rnn", "lstm" or "gru"
    * @param na number of units of the RNN cell
    * @param nx size of the weight matrix multiplying the input
    * @param ny size of the weight matrix relating the hidden-state to the output
    * @return RnnBase of given type
    */
  def apply(rnnType: String, na: Int, nx: Int, ny: Int): RnnBase =
    rnnType match {
      case "rnn"  => RnnBase(RnnCell(na, nx, ny))
      case "lstm" => RnnBase(LstmCell(na, nx, ny))
      case "gru"  => RnnBase(GruCell(na, nx, ny))
      case u      => throw new Error(s"unknown cell type $u")
    }
}
