import Dependencies._

lazy val root = (project in file(".")).settings(
  inThisBuild(
    List(
      organization := "be.botkop",
      scalaVersion := "2.12.5",
      version := "0.1.0-SNAPSHOT"
    )),
  name := "scorch",
  libraryDependencies += numsca,
  libraryDependencies += scalaTest % Test,
  libraryDependencies ++= Seq(
    "com.typesafe.akka" %% "akka-stream" % "2.5.12",
    "com.typesafe.akka" %% "akka-stream-testkit" % "2.5.12" % Test
  )
)

crossScalaVersions := Seq("2.11.12", "2.12.4")

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases" at nexus + "service/local/staging/deploy/maven2")
}

pomIncludeRepository := { _ =>
  false
}

licenses := Seq(
  "BSD-style" -> url("http://www.opensource.org/licenses/bsd-license.php"))

homepage := Some(url("https://github.com/botkop"))

scmInfo := Some(
  ScmInfo(
    url("https://github.com/botkop/scorch"),
    "scm:git@github.com:botkop/scorch.git"
  )
)

developers := List(
  Developer(
    id = "botkop",
    name = "Koen Dejonghe",
    email = "koen@botkop.be",
    url = url("https://github.com/botkop")
  )
)

publishMavenStyle := true
publishArtifact in Test := false
// skip in publish := true
