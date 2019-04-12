package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import HomeWork3.Knn.DistanceCheck;
import HomeWork3.Knn.Weighted;
import weka.core.Instances;

public class MainHW3 {
	private static final int NUM_OF_FOLDS = 10;
	private static final int MAX_NUM_NEIGHBORS = 20;
	private static final double[] LP_VALUES = { 1, 2, 3, Double.POSITIVE_INFINITY };
	private static final int[] NUMBER_OF_FOLDS_LIST = { 0, 50, 10, 5, 3 };

	private static Random random = new Random();
	private static FeatureScaler featureScaler = new FeatureScaler();

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances instancesWithoutFeatureScaling = loadData("auto_price.txt");
		instancesWithoutFeatureScaling.randomize(random);
		
		Instances instancesWithFeatureScaling = featureScaler.scaleData(instancesWithoutFeatureScaling);
		NUMBER_OF_FOLDS_LIST[0] = instancesWithoutFeatureScaling.numInstances();
		
		findBestHyperParameters(instancesWithoutFeatureScaling, false);
		Knn knnBestParametersScaled = findBestHyperParameters(instancesWithFeatureScaling, true);

		for (int numberOfFolds : NUMBER_OF_FOLDS_LIST) {
			System.out.println("----------------------------");
			System.out.println(String.format("Results for %s folds:", numberOfFolds));
			System.out.println("----------------------------");
			
			calculateFold(knnBestParametersScaled, instancesWithFeatureScaling, numberOfFolds, DistanceCheck.Regular);
			System.out.println();
			calculateFold(knnBestParametersScaled, instancesWithFeatureScaling, numberOfFolds, DistanceCheck.Efficient);
		}
	}
	
	private static void calculateFold(Knn knn, Instances instances, int numberOfFolds, DistanceCheck distanceCheck)
			throws Exception {
		knn.setDistanceCheck(DistanceCheck.Regular);
		double error = knn.crossValidationError(instances, numberOfFolds);

		System.out.println(String.format(
				"Cross validation error of %s knn on auto_price dataset is %s and the average elapsed time is %s",
				distanceCheck.getStringValue(), error, knn.getAverageTimeElapse()));
		System.out.println(String.format("The total elapsed time is: %s", knn.getTotalTimeElapse()));
	}

	private static Knn findBestHyperParameters(Instances instances, boolean isDatasetScaled) throws Exception {
		Knn knn = new Knn();
		int minK = -1, minLP = -1, minWeightedDistance = -1;
		double currValidationError, minValidationError = Double.POSITIVE_INFINITY;

		knn.setDistanceCheck(DistanceCheck.Regular);

		for (int i = 1; i <= MAX_NUM_NEIGHBORS; i++) {
			for (int j = 0; j < LP_VALUES.length; j++) {
				for (int k = 0; k < Weighted.values().length; k++) {
					knn.setK(i);
					knn.setLP(LP_VALUES[j]);
					knn.setWeightedDistance(Weighted.values()[k]);

					currValidationError = knn.crossValidationError(instances, NUM_OF_FOLDS);

					if (currValidationError < minValidationError) {
						minK = i;
						minLP = j;
						minWeightedDistance = k;
						minValidationError = currValidationError;
					}
				}
			}
		}

		System.out.println("----------------------------");
		System.out.println("Results for " + (isDatasetScaled ? "scaled" : "original") + " dataset: ");
		System.out.println("----------------------------");
		System.out.println(String.format(
				"Cross validation error with K = %d, lp = %s, majority function = %s for auto_price data is:%s", minK,
				LP_VALUES[minLP], Weighted.values()[minWeightedDistance].getStringValue(), minValidationError));

		knn.setK(minK);
		knn.setLP(LP_VALUES[minLP]);
		knn.setWeightedDistance(Weighted.values()[minWeightedDistance]);

		return knn;
	}
}