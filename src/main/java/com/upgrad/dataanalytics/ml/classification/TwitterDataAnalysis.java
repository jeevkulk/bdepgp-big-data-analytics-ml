package com.upgrad.dataanalytics.ml.classification;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TwitterDataAnalysis {

    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);

        TwitterDataAnalysis twitterDataAnalysis = new TwitterDataAnalysis();
        twitterDataAnalysis.analyzeData();
    }

    public void analyzeData() {
        //*****************Reads data file into a dataset and prints it and it's schema*******************************/
        Dataset<Row> rawDataset = getRowDataset();

        System.out.println("/********************* Rain Forest Model *********************/");
        ClassificationPredictor rainForestClassificationModel = new RainForestClassificationModel();
        rainForestClassificationModel.doDataAnalysis(rawDataset);

        System.out.println("/********************* Decision Tree Model *********************/");
        ClassificationPredictor decisionTreeClassificationModel = new DecisionTreeClassificationModel();
        decisionTreeClassificationModel.doDataAnalysis(rawDataset);
    }

    private Dataset<Row> getRowDataset() {
        SparkSession sparkSession = SparkSession.builder()
                .appName("SparkML")
                .master("local[*]")
                .getOrCreate();

        Dataset<Row> rawDataset =  sparkSession.read()
                .option("header", true)
                .option("inferSchema", true)
                .option("nullValue", true)
                .option("charset", "UTF-8")
                .option("mode", "DROPMALFORMED")
                //.option("multiLine", "false")
                .csv(TwitterDataAnalysis.class.getClassLoader().getResource("data/twitter_gender_classification_data.csv").toString());
        return rawDataset;
    }
}
