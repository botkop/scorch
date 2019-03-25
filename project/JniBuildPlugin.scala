
import sbt._
import sbt.Keys._

import sys.process._


object JniBuildPlugin extends AutoPlugin {

  override val trigger: PluginTrigger = noTrigger

  override val requires: Plugins = plugins.JvmPlugin

  object autoImport extends JniGeneratorKeys {
    lazy val jniBuild = taskKey[Unit]("Builds so lib")
  }

  import autoImport._

  override lazy val projectSettings: Seq[Setting[_]] =Seq(
    
    targetGeneratorDir in jniBuild := sourceDirectory.value / "native" ,

    targetLibName in jniBuild := "java_torch_lib",

    jniBuild := {
      val directory = (targetGeneratorDir in jniBuild).value
      val cmake_prefix = (torchLibPath in jniBuild).value
      val log = streams.value.log

      log.info("Build to " + directory.getAbsolutePath)
      val command = s"cmake -H$directory -B$directory -DCMAKE_PREFIX_PATH=$cmake_prefix"
      log.info(command)
      val exitCode = Process(command) ! log
      if (exitCode != 0) sys.error(s"An error occurred while running cmake. Exit code: $exitCode.")
      val command1 = s"make -C$directory"
      log.info(command1)
      val exitCode1 = Process(command1) ! log
      if (exitCode1 != 0) sys.error(s"An error occurred while running make. Exit code: $exitCode1.")
    },

    jniBuild := jniBuild.dependsOn(jniGen).value,
    compile := (compile in Compile).dependsOn(jniBuild).value,

  )


}
