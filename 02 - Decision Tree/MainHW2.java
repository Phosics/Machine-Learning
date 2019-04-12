package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW2 {
	private static final boolean USE_ENTOPY = false;
	private static final boolean USE_GINI = true;
	private static final int NO_PRUNNING = 0;

	private static final double[] P_VALUES = new double[] { 1, 0.75, 0.5, 0.25, 0.05, 0.005 };

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
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");

		boolean isGini = giniOrEntropy(trainingCancer, validationCancer);
		System.out.println("----------------------------------------------------");

		DecisionTree bestPValueTree = findBestPValue(isGini, trainingCancer, validationCancer);

		System.out.println("Best validation error at p_value = " + P_VALUES[bestPValueTree.getPValueIndex()]);
		System.out.println("Test error with best tree = " + bestPValueTree.calcAvgError(testingCancer));
		System.out.println("Representation of the best tree by ‘if statements’");
		bestPValueTree.printTree();
	}

	private static DecisionTree findBestPValue(boolean isGini, Instances trainingCancer, Instances validationCancer)
			throws Exception {
		DecisionTree pValueTree = null;
		DecisionTree minErrorTree = null;

		double minErrorValue = Double.POSITIVE_INFINITY, currValidationError, currTrainingError;

		for (int i = 0; i < P_VALUES.length; i++) {
			pValueTree = new DecisionTree(isGini, i);
			pValueTree.buildClassifier(trainingCancer);
			currTrainingError = pValueTree.calcAvgError(trainingCancer);
			currValidationError = pValueTree.calcAvgError(validationCancer);
			
			System.out.println("Decision Tree with p_value of: " + P_VALUES[i]);
			System.out.println("The train error of the decision tree is " + currTrainingError);
			System.out.println("Max height on validation data: " + pValueTree.getMaxTreeHeight());
			System.out.println("Average height on validation data: " + pValueTree.getAvgTreeHeight());
			System.out.println("The validation error of the decision tree is " + currValidationError);
			System.out.println("----------------------------------------------------");

			if (currValidationError < minErrorValue) {
				minErrorValue = currValidationError;
				minErrorTree = pValueTree;
			}
		}

		return minErrorTree;
	}

	private static double learnTree(Instances trainingData, Instances validationData, boolean isGini) throws Exception {
		DecisionTree dt = new DecisionTree(isGini, NO_PRUNNING);
		dt.buildClassifier(trainingData);
		
		return dt.calcAvgError(validationData);
	}

	private static boolean giniOrEntropy(Instances trainingData, Instances validationData) throws Exception {
		// Entropy
		double entropyError = learnTree(trainingData, validationData, USE_ENTOPY);
		System.out.println("Validation error using Entropy: " + entropyError);

		// Gini
		double giniError = learnTree(trainingData, validationData, USE_GINI);
		System.out.println("Validation error using Gini: " + giniError);

		return giniError < entropyError;
	}
}
