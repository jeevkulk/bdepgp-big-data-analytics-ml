package com.upgrad.dataanalytics.ml.classification;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import static org.apache.spark.sql.functions.col;

public class DecisionTreeClassificationModel extends ClassificationPredictor {

    Dataset<Row> prepareData(Dataset<Row> rawDataset) {
        Dataset<Row> selectedDataset = rawDataset.select(
                //col("_unit_id").cast("Long"),
                //col("_golden"),
                //col("_unit_state"),
                //col("_trusted_judgments").cast("Integer"),
                //col("_last_judgment_at"),
                col("gender"),
                col("gender:confidence").cast("Double"),
                //col("profile_yn"),
                //col("profile_yn:confidence").cast("Double"),
                //col("profile_yn_gold"),
                //col("created"),
                col("description"),
                col("fav_number").cast("Integer"),
                //col("gender_gold"),
                col("link_color"),
                col("name"),
                //col("profileimage"),
                //col("retweet_count"),
                col("sidebar_color"),
                col("text")
                //col("tweet_coord"),
                //col("tweet_count"),
                //col("tweet_created"),
                //col("tweet_id"),
                //col("tweet_location")
                //col("user_timezone")
        );
        Dataset<Row> filteredDataset = selectedDataset.filter(
                col("gender").contains("male")
                        .or(col("gender").contains("female"))
                        .or(col("gender").contains("brand"))
        );
        filteredDataset = filteredDataset.na().drop();
        /*filteredDataset = filteredDataset.withColumn(
                "verified_gender",
                functions.when(
                        col("_unit_state").equalTo("golden"),
                        col("gender")
                ).otherwise(functions.lit(""))
        );*/
        filteredDataset = filteredDataset.withColumn(
                "feature_str",
                functions.lower(
                functions.concat(
                        col("name"),
                        functions.lit(" "),
                        //col("verified_gender"),
                        //functions.lit(" "),
                        col("description"),
                        functions.lit(" "),
                        col("link_color"),
                        functions.lit(" "),
                        col("sidebar_color"),
                        functions.lit(" "),
                        col("fav_number"),
                        functions.lit(" "),
                        col("text")
                )
                )
        );
        filteredDataset = filteredDataset.withColumn(
                "cleaned_features",
                functions.regexp_replace(col("feature_str"), "[^A-Za-z0-9_ ]", "")
        );
        return filteredDataset;
    }

    Pipeline createModel(Dataset<Row> trainingDataSet) {
        //*************************************String Indexer*********************************************************//
        StringIndexerModel stringIndexerModel = new StringIndexer()
                .setInputCol("gender")
                .setOutputCol("label")
                .fit(trainingDataSet);

        //*************************************Tokenize Data**********************************************************//
        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("cleaned_features")
                .setOutputCol("tokenized_features");

        //*************************************Removing Stop Words***************************************************//
        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("tokenized_features")
                .setOutputCol("filtered_features");

        //*************************************Hashing Term Frequency Matrix******************************************//
        HashingTF hashingTF = new HashingTF()
                .setNumFeatures(100)
                .setInputCol(stopWordsRemover.getOutputCol())
                .setOutputCol("numFeatures");

        //*************************************Hashing Term Frequency Matrix******************************************//
        IDF idf = new IDF()
                .setInputCol(hashingTF.getOutputCol())
                .setOutputCol("features");

        //*************************************Instantiating Decision Tree Classifier*********************************//
        DecisionTreeClassifier decisionTreeClassifier = new DecisionTreeClassifier();

        //*************************************Label Converter********************************************************//
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(stringIndexerModel.labels());

        return new Pipeline()
                .setStages(
                        new PipelineStage[]{
                                stringIndexerModel,
                                tokenizer,
                                stopWordsRemover,
                                hashingTF,
                                idf,
                                decisionTreeClassifier,
                                labelConverter
                        }
                );
    }
}
