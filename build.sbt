import sbt._
import sbt.Keys._


version := "1.0"

scalaVersion := "2.12.7"


// https://mvnrepository.com/artifact/org.bytedeco/javacpp
libraryDependencies += "org.bytedeco" % "javacpp" % "1.4.3"
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.12.7"

enablePlugins(JniGeneratorPlugin, JniBuildPlugin)
JniBuildPlugin.autoImport.torchLibPath in jniBuild := "/home/nazar/libtorch"
//sourceDirectory in nativeCompile := sourceDirectory.value / "native"
//target in nativeCompile :=target.value / "native" / nativePlatform.value


libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.7.2"
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.3"

lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.3"

libraryDependencies += scalaTest % Test


