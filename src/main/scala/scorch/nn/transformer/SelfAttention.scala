package scorch.nn.transformer

import scorch.autograd.Variable
import scorch.nn.{Linear, Module}

case class SelfAttention(emb: Int, heads: Int, mask: Boolean)
    extends Module {

  val toKeys = Linear(emb, emb * heads, useBias = false)
  val toQueries = Linear(emb, emb * heads, useBias = false)
  val toValues = Linear(emb, emb * heads, useBias = false)
  val unifyHeads = Linear(heads * emb, emb)

  override def forward(x: Variable): Variable = {
    val List(b, t, e, h) = x.shape :+ heads

    assert (e == emb, s"Input embedding dim ($e) should match layer embedding dim ($emb)")

    val keys    = toKeys(x).reshape(b, t, h, e) // using reshape iso view here
    val queries = toQueries(x).reshape(b, t, h, e)
    val values  = toValues(x).reshape(b, t, h, e)

    ???
  }

}

object SelfAttention {
}
