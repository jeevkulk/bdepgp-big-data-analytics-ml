package com.upgrad.dataanalytics.ml.classification;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public abstract class ClassificationPredictor {

    public void doDataAnalysis(Dataset<Row> rawDataset) {

        //*************************************Prepare data***********************************************************//
        Dataset<Row> analysisReadyDataset = prepareData(rawDataset);

        //*************************************Dataset Split**********************************************************//
        Dataset<Row>[] datasetSplits = analysisReadyDataset.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingDataset = datasetSplits[0];
        Dataset<Row> testingDataset = datasetSplits[1];

        //*************************************Create Model and Train dataset*****************************************//
        Pipeline pipeline = createModel(trainingDataset);

        PipelineModel pipelineModel = pipeline.fit(trainingDataset);
        Dataset<Row> predictedDataset = pipelineModel.transform(testingDataset);
        predictedDataset = predictedDataset.select(
                "gender",
                "label",
                "features",
                "rawPrediction",
                "probability",
                "prediction",
                "predictedLabel"
        );

        //************************Model Evaluation********************************************************************//

        trainingDataset = pipelineModel.transform(trainingDataset);
        Dataset<Row> predictedTrainingDataset = trainingDataset.select(
                "gender",
                "label",
                "features",
                "rawPrediction",
                "probability",
                "prediction",
                "predictedLabel"
        );

        System.out.println("Evaluation of training data:");
        double trainingAccuracy = evaluateModel(predictedTrainingDataset);
        System.out.println("Evaluation of testing data:");
        double testingAccuracy = evaluateModel(predictedDataset);
        if ((trainingAccuracy - testingAccuracy) / trainingAccuracy < 0.1d) {
            System.out.println("Model is GOOD-FIT");
        } else if (trainingAccuracy - testingAccuracy > 0.25d) {
            System.out.println("Model is OVER-FITTING");
        } else {
            System.out.println("Model is UNDER-FITTING");
        }
    }

    abstract Dataset<Row> prepareData(Dataset<Row> rawDataset);

    abstract Pipeline createModel(Dataset<Row> trainingDataSet);

    private double evaluateModel(Dataset<Row> predictedDataset) {
        predictedDataset = predictedDataset.select("label", "prediction");
        MulticlassMetrics metrics = new MulticlassMetrics(predictedDataset);
        double accuracy = metrics.accuracy();
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion);
        System.out.println("Accuracy = " + metrics.accuracy());
        for (double label : metrics.labels()) {
            System.out.format("Class %f precision = %f\n", label, metrics.precision(label));
            System.out.format("Class %f recall = %f\n", label, metrics.recall(label));
            System.out.format("Class %f F1 score = %f\n", label, metrics.fMeasure(label));
        }
        return accuracy;
    }
}
