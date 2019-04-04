package torch_scala.autograd

import torch_scala.api.aten.{Scalar, Shape, Tensor, TensorType}

import scala.reflect.ClassTag

object MathVariable {

  implicit class MathVariableWrapper[T: ClassTag, TT <: TensorType](self: Variable[T, TT])(implicit num: Numeric[T]) {

    def +(other: Variable[T, TT]): Variable[T, TT] = Add(self, other).forward()
    def -(other: Variable[T, TT]): Variable[T, TT] = Sub(self, other).forward()
    def *(other: Variable[T, TT]): Variable[T, TT] = Mul(self, other).forward()
    def /(other: Variable[T, TT]): Variable[T, TT] = Div(self, other).forward()

    def dot(other: Variable[T, TT]): Variable[T, TT] = Dot(self, other).forward()
    def mm(other: Variable[T, TT]): Variable[T, TT] = Matmul(self, other).forward()

    def unary_- : Variable[T, TT] = Negate(self).forward()
    def +(d: Scalar[T]): Variable[T, TT] = AddConstant(self, d).forward()
    def -(d: Scalar[T]): Variable[T, TT] = SubConstant(self, d).forward()
    def *(d: Scalar[T]): Variable[T, TT] = MulConstant(self, d).forward()
    def /(d: Scalar[T]): Variable[T, TT] = DivConstant(self, d).forward()
    def **(d: Scalar[T]): Variable[T, TT] = PowConstant(self, d).forward()
    def pow(d: Scalar[T]): Variable[T, TT] = PowConstant(self, d).forward()

    def t(): Variable[T, TT] = Transpose(self).forward()
    def reshape(shape: Shape): Variable[T, TT] = Reshape(self, shape).forward()
    def reshape(shape: Int*): Variable[T, TT] = reshape(Shape(shape.toArray))

    def exp(): Variable[T, TT] = Exp(self).forward()
    def cos(): Variable[T, TT] = Cos(self).forward()
    def sin(): Variable[T, TT] = Sin(self).forward()
    def mean(): Variable[T, TT] = Mean(self).forward()
    def mean(axis: Int): Variable[T, TT] = MeanByAxis(self, axis).forward()
    def sigmoid(): Variable[T, TT] = Sigmoid(self).forward()
    // def softmax(dim: Long): Variable[T, TT] = Softmax(self, dim).forward()
    def tanh(): Variable[T, TT] = Tanh(self).forward()
    def relu(): Variable[T, TT] = Threshold(self, num.zero).forward()
    def variance(): Variable[T, TT] = Variance(self).forward()
    def variance(axis: Int): Variable[T, TT] = VarianceByAxis(self, axis).forward()
    def sqrt(): Variable[T, TT] = Sqrt(self).forward()
    def abs(): Variable[T, TT] = Abs(self).forward()

  }

  implicit class MathScalarWrapper[T: ClassTag](self: Scalar[T])(implicit num: Numeric[T]) {
    def +[TT <: TensorType](t: Variable[T, TT]): Variable[T, TT] = t + self
    def -[TT <: TensorType](t: Variable[T, TT]): Variable[T, TT] = -t + self
    def *[TT <: TensorType](t: Variable[T, TT]): Variable[T, TT] = t * self
    def /[TT <: TensorType](t: Variable[T, TT]): Variable[T, TT] = t.**(num.fromInt(-1)) * self

  }

  implicit class MathDoubleWrapper(self: Double) {
    def +[TT <: TensorType](t: Variable[Double, TT]): Variable[Double, TT] = t + self
    def -[TT <: TensorType](t: Variable[Double, TT]): Variable[Double, TT] = -t + self
    def *[TT <: TensorType](t: Variable[Double, TT]): Variable[Double, TT] = t * self
    def /[TT <: TensorType](t: Variable[Double, TT]): Variable[Double, TT] = t.**(-1) * self

  }

}
