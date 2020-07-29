
name := "main/scala/test"


version:= "1.0"
scalaVersion := "2.12.10"

lazy val root = (project in file("project"))

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" %  "3.0.0-preview2",
    "org.apache.spark" %% "spark-sql" %  "3.0.0-preview2"
)
