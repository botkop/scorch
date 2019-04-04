
import java.io.{File, FileInputStream}

import org.objectweb.asm.{ClassReader, ClassVisitor, MethodVisitor, Opcodes}

import scala.collection.JavaConverters._
import scala.collection.mutable
import sbt._
import sbt.Keys._

import sys.process._

trait JniGeneratorKeys {

  lazy val torchLibPath = settingKey[String]("Path to C++ torch library.")

  lazy val targetGeneratorDir = settingKey[File]("target directory to store generated cpp files.")

  lazy val targetLibName = settingKey[String]("target cpp file name.")

  lazy val builderClass = settingKey[String]("class name that generates cpp file.")

  lazy val jniGen = taskKey[Unit]("Generates cpp files")

  lazy val javahClasses: TaskKey[Set[String]] = taskKey[Set[String]](
    "Finds the fully qualified names of classes containing native declarations.")

}


object JniGeneratorPlugin extends AutoPlugin {

  override val trigger: PluginTrigger = noTrigger

  override val requires: Plugins = plugins.JvmPlugin

  object autoImport extends JniGeneratorKeys

  import autoImport._

  override lazy val projectSettings: Seq[Setting[_]] =Seq(
    javahClasses in jniGen := {
      import xsbti.compile._
      val compiled: CompileAnalysis = (compile in Compile).value
      val classFiles: Set[File] = compiled.readStamps.getAllProductStamps.asScala.keySet.toSet
      val nativeClasses = classFiles flatMap { file => findNativeClasses(file) }
      nativeClasses
    },

    targetGeneratorDir in jniGen := sourceDirectory.value / "native" ,

    targetLibName in jniGen := "java_torch_lib",

    builderClass in jniGen := "generate.Builder",

    jniGen := {
      val directory = (targetGeneratorDir in jniGen).value
      val builder = (builderClass in jniGen).value
      val libName = (targetLibName in jniGen).value
      // The full classpath cannot be used here since it also generates resources. In a project combining JniJavah and
      // JniPackage, we would have a chicken-and-egg problem.
      val classPath: String = ((dependencyClasspath in Compile).value.map(_.data) ++ {
        Seq((classDirectory in Compile).value)
      }).mkString(sys.props("path.separator"))
      val classes = (javahClasses in jniGen).value
      val log = streams.value.log

      if (classes.nonEmpty) {
        log.info("Sources will be generated to " + directory.getAbsolutePath)
        log.info("Generating header for " + classes.mkString(" "))
        val command = s"java -classpath $classPath $builder -d $directory -o  $libName ${classes.mkString(" ")}" // " torch_scala.NativeLibraryConfig" }"
        log.info(command)
        val exitCode = Process(command) ! log
        if (exitCode != 0) sys.error(s"An error occurred while running javah. Exit code: $exitCode.")
      }
    }

  )

  private class NativeFinder extends ClassVisitor(Opcodes.ASM5) {
    private var fullyQualifiedName: String = ""

    /** Classes found to contain at least one @native definition. */
    private val _nativeClasses = mutable.HashSet.empty[String]

    def nativeClasses: Set[String] = _nativeClasses.toSet

    override def visit(
                        version: Int, access: Int, name: String, signature: String, superName: String,
                        interfaces: Array[String]): Unit = {
      fullyQualifiedName = name.replaceAll("/", ".")
    }

    override def visitMethod(
                              access: Int, name: String, desc: String, signature: String, exceptions: Array[String]): MethodVisitor = {
      val isNative = (access & Opcodes.ACC_NATIVE) != 0
      if (isNative)
        _nativeClasses += fullyQualifiedName
      // Return null, meaning that we do not want to visit the method further.
      null
    }
  }

  /** Finds classes containing native implementations (i.e., `@native` definitions).
    *
    * @param  javaFile Java file from which classes are being read.
    * @return Set containing all the fully qualified names of classes that contain at least one member annotated with
    *         the `@native` annotation.
    */
  def findNativeClasses(javaFile: File): Set[String] = {
    var inputStream: FileInputStream = null
    try {
      inputStream = new FileInputStream(javaFile)
      val reader = new ClassReader(inputStream)
      val finder = new NativeFinder
      reader.accept(finder, 0)
      finder.nativeClasses
    } finally {
      if (inputStream != null)
        inputStream.close()
    }
  }


}
