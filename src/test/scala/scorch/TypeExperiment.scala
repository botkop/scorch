package scorch

import botkop.numsca.Tensor
import org.scalatest.{FlatSpec, Matchers}
import scorch.autograd.Variable

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe._

abstract class AbstractModule
case class Zap() extends AbstractModule

class TypeExperiment extends FlatSpec with Matchers {

  "A Zoop" should "find Zaps" in {

    abstract class Zoop() {
      def zaps: List[AbstractModule] = {
        this.getClass.getDeclaredFields.toList.flatMap { f =>
          f setAccessible true
          val v: AnyRef = f.get(this)
          println(v.getClass.getName)
          println(v.isInstanceOf[AbstractModule])
          v match {
            case module: AbstractModule => Some(module)
            case _                      => None
          }

        }
      }
    }

    case class A(x: AbstractModule) extends Zoop {
      val a = Zap()
      val b = Zap()
    }

    val x = Zap()
    val za = A(x)
    val zaps = za.zaps
    println(zaps)

  }

}

object TypeExperiment extends App {

  type Parameter = Variable

  abstract class AbstractModule(localParameters: Seq[Parameter] = Nil) {
    def parameters: Unit = {
      this.getClass.getDeclaredFields.toSeq.foreach { f =>
        println(f.getType)

        f setAccessible true
        f.get(this) match {
          case p @ TypeRef(a: universe.Type,
                           sym: universe.Symbol,
                           args: List[universe.Type]) =>
            println(a)
            println(sym)
            Some(p)
          case _ => None
        }
      // ???
      }
    }
  }

  case class MyModule() extends AbstractModule {
    val p = new Parameter(Tensor(3))
    val v = Variable(Tensor(7))
    val a = 10
  }

  val m = MyModule()
  println(m.parameters)

}
