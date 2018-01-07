import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "be.botkop",
      scalaVersion := "2.12.4",
      version      := "0.1.1-SNAPSHOT"
    )),
    name := "autograd",
    libraryDependencies += numsca,
    libraryDependencies += scalaTest % Test
  )
