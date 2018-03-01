package scorch.sandbox.rnn.word

import botkop.{numsca => ns}
import scorch.autograd.Variable
import scorch.nn.rnn.{RnnCellBase, GruCell, LstmCell, RnnCell}
import scorch.nn.{Adam, Optimizer, SGD, _}
import scorch._

import scala.io.Source
import scala.util.Random

class WordModel(corpus: Seq[String],
                tokenizer: (String) => Seq[String],
                eosSymbol: String = "<EOS>",
                cellType: String = "gru",
                optimizerType: String = "adam",
                learningRate: Double = 0.06,
                na: Int = 50,
                numIterations: Int = Int.MaxValue,
                maxSentenceSize: Int = 60,
                numSentences: Int = 8,
                printEvery: Int = 100,
) {

  val tokens: Seq[String] =
    tokenizer(corpus.mkString(" ")).distinct.sorted :+ eosSymbol

  val vocabSize: Int = tokens.length
  val tokenToIdx: Map[String, Int] = tokens.zipWithIndex.toMap

  val bosIndex: Int = -1 // index for beginning of sentence
  val eosIndex: Int = tokenToIdx(eosSymbol) // index for end of sentence
  val (nx, ny) = (vocabSize, vocabSize)

  // define the RNN model
  val rnn: RnnCellBase = cellType match {
    case "rnn"  => RnnCell(na, nx, ny)
    case "lstm" => LstmCell(na, nx, ny)
    case "gru"  => GruCell(na, nx, ny)
    case u      => throw new Error(s"unknown cell type $u")
  }

  val optimizer: Optimizer = optimizerType match {
    case "sgd"      => SGD(rnn.parameters, learningRate)
    case "adam"     => Adam(rnn.parameters, learningRate)
    case "nesterov" => Nesterov(rnn.parameters, learningRate)
    case u          => throw new Error(s"unknown optimizer type $u")
  }

  val sampler = WordSampler(rnn, tokenToIdx, eosIndex, maxSentenceSize)

  def run(): Unit = {
    var totalLoss = 0.0
    for (j <- 1 to numIterations) {
      val index = j % corpus.length

      val xs: Seq[Int] = bosIndex +: tokenizer(corpus(index)).map(tokenToIdx)
      val ys: Seq[Int] = xs.tail :+ eosIndex

      val loss = optimize(xs, ys)
      totalLoss += loss

      if (j % printEvery == 0) {
        println(s"Iteration: $j, Loss: ${totalLoss / printEvery}")
        (1 to numSentences).foreach(_ =>
          println(sampler.sample((ls: Seq[String]) => ls.mkString(" "))))
        println()
        totalLoss = 0.0
      }
    }
  }

  /**
    * Execute one step of the optimization to train the model
    * @param xs list of integers, where each integer is a number that maps to a token in the vocabulary
    * @param ys list of integers, exactly the same as xs but shifted one index to the left
    * @return the value of the loss function (cross-entropy)
    */
  def optimize(xs: Seq[Int], ys: Seq[Int]): Double = {
    optimizer.zeroGrad()
    val yHat = rnnForward(xs)
    val loss = crossEntropyLoss(yHat, ys)
    loss.backward()
    optimizer.step()
    loss.data.squeeze()
  }

  /**
    * Performs the forward propagation through the RNN
    * @param xs sequence of input tokens to activate
    * @return predictions of the RNN over xs
    */
  def rnnForward(xs: Seq[Int]): Seq[Variable] =
    xs.foldLeft(List.empty[Variable], rnn.initialTrackingStates) {
        case ((yhs, p0), x) =>
          // one hot encoding of x
          val xt = Variable(ns.zeros(vocabSize, 1))
          if (x != bosIndex)
            xt.data(x, 0) := 1

          val next = rnn(xt +: p0: _*)
          val (yht, p1) = (next.head, next.tail)
          (yhs :+ yht, p1)
      }
      ._1

}

object WordModel {
  def apply(fileName: String,
            tokenizer: (String) => Seq[String] = (s) => s.split("\\s+"),
            eosSymbol: String = "<EOS>",
            cellType: String = "gru",
            optimizerType: String = "adam",
            learningRate: Double = 0.06,
            na: Int = 50,
            numIterations: Int = Int.MaxValue,
            maxSentenceSize: Int = 60,
            numSentences: Int = 8,
            printEvery: Int = 100,
  ): WordModel = {
    val corpus = Random.shuffle(
      Source
        .fromFile(fileName)
        .getLines()
        .toList)
    new WordModel(corpus,
                  tokenizer,
                  eosSymbol,
                  cellType,
                  optimizerType,
                  learningRate,
                  na,
                  numIterations,
                  maxSentenceSize,
                  numSentences,
                  printEvery)
  }

  def main(args: Array[String]): Unit = {
    //     val fname = args.head
    val fname = "src/test/resources/sonnets-cleaned.txt"
    WordModel(fname,
              tokenizer = (s) =>
                s.toLowerCase
                  .replaceAll("[\\.',;:\\-!\\?\\(]+", " ")
                  .split("\\s+")
                  .filterNot(_.isEmpty),
              learningRate = 0.001,
              printEvery = 1000)
      .run()
  }

}
