
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.sql.functions._
import org.apache.spark.sql
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.Pipeline

object main {

  val csv_structure = new StructType()
                        .add("user_id",LongType)
                        .add("channel",StringType)
                        .add("program_name",StringType)
                        .add("normal_program_name",StringType)
                        .add("program_theme",StringType)
                        .add("program_subtheme",StringType)
                        .add("date_time_start",TimestampType)
                        .add("end_date_time",TimestampType)
                        .add("program_start",TimestampType)
                        .add("program_end",TimestampType)
                        .add("real_start_time",TimestampType)
                        .add("real_session_duration",IntegerType)


  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
                .appName("ModeloTV")
                .master("local")
                .getOrCreate()

    import spark.implicits._

    val csv = spark
                .read
                .format("com.databricks.spark.csv")
                .schema(csv_structure)
                .load("data/data.csv")

    val filtrado = filtrarExtremos(csv)
    val clean_scaledData = estandarizarFeatures(filtrado)
    val filtrado_agregado_Data = dataAgregada(clean_scaledData );

    //Selecciono lo que me sirve
    val clean_df = filtrado_agregado_Data.select("norm_real_session_duration","manana","mediodia","tarde","noche","program_theme","program_subtheme")

    //Clasifico en contextos (Primero Theme, Segundo subtheme)
    val theme_df = clean_df.select("norm_real_session_duration","manana","mediodia","tarde","noche","program_theme")

    val predictions = clasifico_theme(theme_df);

    // Select (prediction, true label) and compute test error.
    evaluo(predictions,"label")

    //Random Forest
    val predictions_rf = clasifico_theme_rf(theme_df);
    evaluo(predictions_rf,"indexedLabel")


  }

  def filtrarExtremos(df: DataFrame): DataFrame = {
    //Filtro los extremos
    return df.filter("real_session_duration > 10") .filter("real_session_duration < 10800")
  }

  def estandarizarFeatures(df: DataFrame): DataFrame = {
    //Estandarizo algunas features
    val toArray = udf((x: Int) => org.apache.spark.ml.linalg.Vectors.dense(x))
    val vect_clean = df.withColumn("real_session_duration", toArray(col("real_session_duration")))

    val scaler = new MinMaxScaler().setInputCol("real_session_duration").setOutputCol("norm_real_session_duration")

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(vect_clean)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(vect_clean)

    //Paso el feature de vector a double
    val toDouble = udf((x: org.apache.spark.ml.linalg.Vector) => x(0))
    return scaledData.withColumn("norm_real_session_duration", toDouble(col("norm_real_session_duration")))
  }

  def dataAgregada(df: DataFrame): DataFrame = {
    df.
      withColumn("manana", when(hour(col("program_start")) > 4 && hour(col("program_start")) < 10, 1).otherwise(0)).
      withColumn("mediodia", when(hour(col("program_start")) >= 10 && hour(col("program_start")) <= 14, 1).otherwise(0)).
      withColumn("tarde",when(hour(col("program_start")) >= 14 && hour(col("program_start")) <= 18, 1).otherwise(0)).
      withColumn("noche", when(hour(col("program_start")) >= 18 || hour(col("program_start")) <= 4, 1).otherwise(0))
  }

  def clasifico_theme(theme_df: DataFrame): DataFrame = {

    val Array(trainingData, testData) = theme_df.randomSplit(Array(0.7, 0.3))


    //Miro tod0 el dataset asi conozco todos los labels posibles.
    val labelIndexer = new StringIndexer().
      setInputCol("program_theme").
      setOutputCol("label").
      fit(theme_df)


    val features_assembler = new VectorAssembler()
      .setInputCols(Array("norm_real_session_duration","manana","mediodia","tarde","noche"))
      .setOutputCol("features")

    val trainig_set = features_assembler.transform(trainingData)

    val lr = new LogisticRegression().
      setMaxIter(10).
      setRegParam(0.3).
      setElasticNetParam(0.8).
      setFeaturesCol("features")

    val pipeline = new Pipeline().setStages(Array(labelIndexer, features_assembler, lr))

    //Entreno el modelo
    val model = pipeline.fit(trainingData)
    // Make predictions.
    return model.transform(testData)

  }

  def evaluo(predictions: DataFrame, label: String): Unit = {

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol(label).
      setPredictionCol("prediction").
      setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Acc. = " + (accuracy))

  }

  def clasifico_theme_rf(theme_df: DataFrame): DataFrame = {

    val labelIndexer = new StringIndexer().
      setInputCol("program_theme").
      setOutputCol("indexedLabel").
      fit(theme_df)

    val features_assembler = new org.apache.spark.ml.feature.VectorAssembler().
      setInputCols(Array("norm_real_session_duration","manana","mediodia","tarde","noche")).
      setOutputCol("features")

    val theme_assembled = features_assembler.transform(theme_df)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer().
      setInputCol("features").
      setOutputCol("indexedFeatures").
      setMaxCategories(4).
      fit(theme_assembled)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = theme_df.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier().
      setLabelCol("indexedLabel").
      setFeaturesCol("indexedFeatures").
      setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().
      setInputCol("prediction").
      setOutputCol("predictedLabel").
      setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline().
      setStages(Array(labelIndexer, features_assembler, featureIndexer, rf, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    return model.transform(testData)
  }
}
