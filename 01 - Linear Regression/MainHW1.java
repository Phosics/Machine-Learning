package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW1 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	/**
	 * Sets the class index as the last attribute.
	 * 
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances trainingData = loadData(args[0]);
		Instances testData = loadData(args[1]);
		Instances trainingTrio, testTrio;

		// Find best alpha and build classifier with all attributes
		LinearRegression lr = new LinearRegression(trainingData);
		System.out.println("The chosen alpha is: " + lr.getAlpha());
		System.out.println("Training error with all features is: " + lr.calculateMSE(trainingData));
		System.out.println("Test error with all features is: " + lr.calculateMSE(testData));
		System.out.println();

		System.out.println("List of all combination of 3 features");
		System.out.println("-------------------------------------");

		// build classifiers with all 3 attributes combinations
		double minTrioError = Double.POSITIVE_INFINITY;
		int[] trio = new int[3];

		for (int i = 0; i < trainingData.numAttributes() - 1; i++) {
			for (int j = i + 1; j < trainingData.numAttributes() - 1; j++) {
				for (int k = j + 1; k < trainingData.numAttributes() - 1; k++) {
					trainingTrio = loadData(args[0]);
					double trioError = trainTrio(i, j, k, loadData(args[0]), lr);

					System.out.println(i + "-" + j + "-" + k + " " + trioError);

					if (trioError < minTrioError) {
						minTrioError = trioError;
						trio[0] = i;
						trio[1] = j;
						trio[2] = k;
					}
				}
			}
		}

		trainingTrio = loadData(args[0]);
		testTrio = loadData(args[1]);
		removeUnusedAttributes(trio[0], trio[1], trio[2], testTrio);

		trainTrio(trio[0], trio[1], trio[2], trainingTrio, lr);
		System.out.println(
				"Training error the features " + trio[0] + "-" + trio[1] + "-" + trio[2] + ": " + minTrioError);
		System.out.println("Test error the features " + trio[0] + "-" + trio[1] + "-" + trio[2] + ": "
				+ lr.calculateMSE(testTrio));
	}

	private static double trainTrio(int i, int j, int k, Instances trainingTrio, LinearRegression lr) throws Exception {
		removeUnusedAttributes(i, j, k, trainingTrio);

		// Training the data.
		lr.buildClassifier(trainingTrio);
		return lr.calculateMSE(trainingTrio);
	}

	private static void removeUnusedAttributes(int i, int j, int k, Instances data) {
		for (int l = data.numAttributes() - 2; l >= 0; l--) {
			if (l != i && l != j && l != k) {
				data.deleteAttributeAt(l);
			}
		}
	}
}
