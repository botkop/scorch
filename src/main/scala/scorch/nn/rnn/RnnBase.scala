package scorch.nn.rnn

import scorch.autograd.Variable
import scorch.nn.MultiVarModule

case class RnnBase(cell: RnnCellBase) extends MultiVarModule(cell.parameters) {

  /**
    * Performs the forward propagation through the RNN
    * @param xs sequence of variables to activate
    * @return predictions of the RNN over xs
    */
  override def forward(xs: Seq[Variable]): Seq[Variable] =
    xs.foldLeft(List.empty[Variable], cell.initialTrackingStates) {
        case ((yhs, p0), x) =>
          val next = cell(x +: p0: _*)
          val (yht, p1) = (next.head, next.tail)
          (yhs :+ yht, p1)
      }
      ._1
}

object RnnBase {
  def apply(rnnType: String, na: Int, nx: Int, ny: Int): RnnBase =
    rnnType match {
      case "rnn"  => RnnBase(RnnCell(na, nx, ny))
      case "lstm" => RnnBase(LstmCell(na, nx, ny))
      case "gru"  => RnnBase(GruCell(na, nx, ny))
      case u      => throw new Error(s"unknown cell type $u")
    }
}
