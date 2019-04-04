package torch_scala.api.aten

import torch_scala.api.exception.InvalidShapeException

final class Shape private (private val array: Array[Int]) {
  /** Returns a boolean value indicating whether this shape is fully defined.
    *
    * If the size of any dimension is equal to `-1` or if the shape is completely unknown, then it is not considered
    * fully defined..
    */
  def isFullyDefined: Boolean = array != null && !array.contains(-1)

  /** Gets the rank of this shape (i.e., number of dimensions). */
  def rank: Int = if (array == null) -1 else array.length

  /** Gets the size for a specific dimension of this shape.
    *
    * @param  dimension Dimension whose size to return.
    */
  def size(dimension: Int): Int = {
    if (dimension < 0)
      array(array.length + dimension)
    else
      array(dimension)
  }

  /** Gets the total number of elements in tensors of this shape.
    *
    * If the shape is not fully defined, then `-1` is returned, otherwise, the product of the sizes for each dimension
    * of this shape is returned.
    */
  def numElements: Long = {
    if (!isFullyDefined) {
      -1
    } else {
      var size: Long = 1L
      array.foreach(size *= _)
      size
    }
  }

  /** Reshapes this shape to the provided shape.
    *
    * This function first checks that this shape can be reshaped in the specified way and then:
    *   - If `shape` has an unknown dimension, then its value is computed, filled in, and the new shape is returned.
    *   - Otherwise, `shape` is returned.
    *
    * @param  shape Shape to reshape to.
    * @return New shape.
    * @throws IllegalArgumentException If this shape cannot be reshaped to `shape`.
    */
  @throws[IllegalArgumentException]
  def reshape(shape: Shape): Shape = {
    this.assertFullyDefined("Only fully defined shapes can be reshaped.")
    val unknownDimensions = shape.asArray.count(_ == -1)
    if (shape.rank == -1 || unknownDimensions > 1)
      throw new IllegalArgumentException(
        s"The new shape ($shape) must have known rank and at most one unknown dimension.")
    if (unknownDimensions == 0 && this.numElements != shape.numElements) {
      throw new IllegalArgumentException(
        s"Shape '$this' cannot be reshaped to '$shape' (different number of elements).")
    } else if (unknownDimensions == 0) {
      shape
    } else {
      val unknownIndex = shape.asArray.indexWhere(_ == -1)
      val otherNumElements = shape.asArray.filter(_ == -1).product
      if (this.numElements % otherNumElements != 0)
        throw new IllegalArgumentException(s"Shape '$this' cannot be reshaped to '$shape'.")
      val newShape = shape.asArray
      newShape(unknownIndex) = (this.numElements / otherNumElements).toInt
      new Shape(newShape)
    }
  }

  /** Returns an array representation of this shape. This method does not perform a copy or an array creation. It simply
    * returns the underlying array representation of this shape. Its cost is thus the same as that of a field access. */
  def asArray: Array[Int] = array

  /** Checks if `other` is compatible with this shape.
    *
    * Two shapes are compatible if either of them is completely unknown, or if they have the same rank and each one of
    * their dimensions is compatible. One dimension is compatible with another if either one is equal to `-1` or if they
    * have the same value.
    *
    * For example:
    *  - `Shape.unknown()` is compatible with every other shape.
    *  - `Shape.unknown(rank = r)` is compatible with every other shape which has rank `r`.
    *  - `Shape(-1, -1)` is compatible with all rank `2` shapes.
    *  - `Shape(32, -1)` is compatible with all rank `2` shapes whose first dimension size is equal to `-1` or `32`.
    *  - `Shape(32, 784)` is compatible with itself and `Shape(-1, 784)`, `Shape(32, -1)`, `Shape.unknown(rank = 2)`,
    * and `Shape.unknown()`.
    *
    * The compatibility relation is reflexive and symmetric, but not transitive. For example, `Shape(32, 784)` is
    * compatible with `Shape.unknown()`, and `Shape.unknown()` is compatible with `Shape(4, 4)`, but `Shape(32, 784)` is
    * not compatible with `Shape(4, 4)`.
    *
    * @param  other Shape to check compatibility with.
    * @return Boolean value indicating whether the two shapes are compatible.
    */
  def isCompatibleWith(other: Shape): Boolean = {
    this.rank == -1 || other.rank == -1 ||
      (this.rank == other.rank &&
        this.array != null && other.array != null &&
        this.array.zip(other.asArray).forall(t => t._1 == -1 || t._2 == -1 || t._1 == t._2))
  }

  /** Merges two shapes and returns the result as a new shape.
    *
    * Merging consists of first checking whether the shapes are compatible using the [[isCompatibleWith]] method and
    * then going through each dimension of this shape and keeping it if it not equal to `-1` (i.e., unknown), or setting
    * it equal to the respective dimension of `other`, otherwise. This effectively merges the information contained in
    * the two shapes.
    *
    * For example:
    * {{{
    *   val shape1 = Shape(2, 3, -1, 1)
    *   val shape2 = Shape(-1, 3, 5, -1)
    *   val mergedShape = shape1.mergeWith(shape2)
    *   assert(mergedShape == Shape(2, 3, 5, 1)
    * }}}
    *
    * The merging functionality is reflexive and symmetric, but not transitive, similar to the compatibility relation.
    *
    * @param  other Shape to merge with.
    * @throws InvalidShapeException If this shape is not compatible with `other`.
    */
  @throws[InvalidShapeException]
  def mergeWith(other: Shape): Shape = {
    if (this.rank == -1) {
      other
    } else if (other.rank == -1) {
      this
    } else {
      assertSameRank(other)
      assertIsCompatibleWith(other)
      new Shape(this.array.zip(other.asArray).map(t => {
        if (t._1 == -1) t._2
        else if (t._2 == -1) t._1
        else t._1
      }))
    }
  }

  def +(dimension: Int): Shape = new Shape(this.array :+ dimension)
  def ++(other: Shape): Shape = concatenateWith(other)

  // TODO: Support merging an unknown shape with a (partially) known one and vice-versa.
  /** Concatenates this shape with another shape and returns the result as a new shape.
    *
    * If any of the two shapes is completely unknown, then the result of the concatenation is also a completely unknown
    * shape. Otherwise, the two shapes are simply concatenated.
    *
    * For example:
    * {{{
    *   val shape1 = Shape(2, 3, -1, 1)
    *   val shape2 = Shape(-1, 3, 5, -1)
    *   val shape3 = Shape.unknown()
    *   val shape12 = shape1.concatenateWith(shape2)
    *   assert(shape12 == Shape(2, 3, -1, 1, -1, 3, 5, -1)
    *   val shape23 = shape2.concatenateWith(shape3)
    *   assert(shape23 == Shape.unknown())
    *   val shape31 = shape3.concatenateWith(shape1)
    *   assert(shape31 == Shape.unknown())
    * }}}
    *
    * @param  other Shape to concatenate with this shape.
    */
  def concatenateWith(other: Shape): Shape = {
    if (this.rank == -1 || other.rank == -1)
      new Shape(null)
    else
      new Shape(this.array ++ other.array)
  }

  /** Returns a shape with the specified rank that is based on the current shape.
    *
    * This method can be used to promote a completely unknown shape to one with a known rank.
    *
    * @param  rank Rank to use for the new shape.
    * @throws InvalidShapeException If this shape is fully or partially known and has a different rank than the
    *                               provided value.
    */
  @throws[InvalidShapeException]
  def withRank(rank: Int): Shape = mergeWith(Shape.unknown(rank))

  /** Returns a shape with at least the specified rank, that is based on the current shape.
    *
    * @param  rank Minimum rank to use for the new shape.
    * @throws InvalidShapeException If this shape is fully or partially known and has a rank that is smaller than the
    *                               provided value.
    */
  @throws[InvalidShapeException]
  def withRankAtLeast(rank: Int): Shape = {
    assertRankAtLeast(rank)
    this
  }

  /** Asserts that this shape is fully defined (i.e., fully known). If it is not, an [[InvalidShapeException]] exception
    * is thrown.
    *
    * @throws InvalidShapeException If this shape is not fully defined.
    */
  @throws[InvalidShapeException]
  def assertFullyDefined(message: String = s"Shape '$this' must be fully defined."): Unit = {
    if (!this.isFullyDefined)
      throw InvalidShapeException(message)
  }

  /** Asserts that this shape has the specified rank.
    *
    * @param  rank Rank.
    * @throws InvalidShapeException If this shape has rank other than `rank`.
    */
  @throws[InvalidShapeException]
  def assertHasRank(rank: Int): Unit = {
    if (this.rank != -1 && this.rank != rank)
      throw InvalidShapeException(s"Shape '$this' must have rank $rank.")
  }

  /** Asserts that this shape has rank at least `rank` and throws an exception if it does not.
    *
    * @param  rank Rank lower bound.
    * @throws InvalidShapeException If this shape has rank lower than `rank`.
    */
  @throws[InvalidShapeException]
  def assertRankAtLeast(rank: Int): Unit = {
    if (this.rank < rank)
      throw InvalidShapeException(s"Shape '$this' must have rank at least $rank.")
  }

  /** Asserts that this shape has rank at most `rank` and throws an exception if it does not.
    *
    * @param  rank Rank upper bound.
    * @throws InvalidShapeException If this shape has rank higher than `rank`.
    */
  @throws[InvalidShapeException]
  def assertRankAtMost(rank: Int): Unit = {
    if (this.rank > rank)
      throw InvalidShapeException(s"Shape '$this' must have rank at most $rank.")
  }

  /** Asserts that this shape has the same rank as `other`. If the two shapes are not compatible, an
    * [[InvalidShapeException]] exception is thrown.
    *
    * @param  other Shape to assert having the same rank as.
    * @throws InvalidShapeException If this shape does not have the same rank as `other`.
    */
  @throws[InvalidShapeException]
  def assertSameRank(other: Shape): Unit = {
    if (this.rank != other.rank)
      throw InvalidShapeException(s"Shape '$this' must have the same rank as shape '$other'.")
  }

  /** Asserts that this shape is compatible with `other` using the [[isCompatibleWith]] method. If the two shapes are
    * not compatible, an [[InvalidShapeException]] exception is thrown.
    *
    * This method can be used to assert that there exists a shape that both this shape and `other` represent.
    *
    * @param  other Shape to assert compatibility with.
    * @throws InvalidShapeException If this shape is not compatible with `other`.
    */
  @throws[InvalidShapeException]
  def assertIsCompatibleWith(other: Shape): Unit = {
    if (!isCompatibleWith(other))
      throw InvalidShapeException(s"Shape '$this' must be compatible with shape '$other'.")
  }

  /** Gets the size for a specific dimension of this shape.
    *
    * @param  dimension Dimension whose size to return.
    */
  def apply(dimension: Int): Int = {
    size(dimension)
  }

  /** Gets a slice of this shape.
    *
    * @param  slice Slice to get.
    */
  def apply(slice: Slice): Shape = {
    if (slice == null)
      throw new IllegalArgumentException("The provided slice should not be 'null'.")
    if (array != null)
      Shape.fromSeq(slice.toArray(rank).map(i => array(i)))
    else
      Shape.unknown(slice.length(rank))
  }


  override def toString: String = {
    if (array == null) {
      "<unknown>"
    } else {
      s"[${array.mkString(", ").replace("-1", "?")}]"
    }
  }

  override def equals(that: Any): Boolean = that match {
    case that: Shape =>
      if ((this.rank != that.rank)
        || (this.array == null && that.array != null)
        || (this.array != null && that.array == null))
        false
      else if (this.array == null && that.array == null)
        true
      else
        this.array.sameElements(that.array)
    case _ => false
  }

  override def hashCode: Int = array.hashCode

  def zip(other: Shape): Array[(Int, Int)] = {
    val max_rank = math.max(rank, other.rank)
    Array.range(0, max_rank).map(i => {
      val ai = if(rank > i) array(i) else 1
      val bi = if(other.rank > i) other(i) else 1
      (ai, bi)
    })
  }

  def isBroadcastableTo(other: Shape): Boolean = {
    zip(other).forall({case (si, oi) => si == 1 || si == oi})
  }
}

/** Contains helper functions for creating [[Shape]] objects. */
object Shape {
  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def create(dimensions: Int*): Shape = new Shape(Array(dimensions: _*))

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def create(dimensions: Array[Int]): Shape = new Shape(dimensions)

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def fromSeq(dimensions: Seq[Int]): Shape = new Shape(dimensions.toArray)

  /** Creates an unknown shape, optionally with a known rank.
    *
    * @param  rank Optional rank of the shape to create. If set to `-1`, then it is considered unknown.
    */
  def unknown(rank: Int = -1): Shape = if (rank == -1) new Shape(null) else new Shape(Array.fill[Int](rank)(-1))

  /** Creates a shape representing a scalar. */
  def scalar(): Shape = Shape.create()

  /** Creates a shape representing a vector with the specified length.
    *
    * @param  length Vector length.
    */
  def vector(length: Int): Shape = Shape.create(length)

  /** Creates a shape representing a matrix with the specified number of rows and columns.
    *
    * @param  numRows    Matrix number of rows.
    * @param  numColumns Matrix number of columns.
    */
  def matrix(numRows: Int, numColumns: Int): Shape = Shape.create(numRows, numColumns)

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def apply(dimensions: Int*): Shape = create(dimensions: _*)

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def apply(dimensions: Array[Int]): Shape = create(dimensions)


}