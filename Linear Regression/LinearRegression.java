package HomeWork1;

import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	private static final double STOPPING_GD = 0.003;
	private static final int ITERATION_LIMIT = 20000;

	private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;

	/**
	 * Building new LinearRegression. Finding the best alpha. Training it on the
	 * given dataset.
	 * 
	 * @param trainingData
	 *            The data to train
	 * @throws Exception
	 */
	public LinearRegression(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1;

		findAlpha(trainingData);
		m_coefficients = gradientDescent(trainingData);
	}
	
	// the method which runs to train the linear regression predictor, i.e.
	// finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		m_truNumAttributes = trainingData.numAttributes() - 1;
		m_coefficients = gradientDescent(trainingData);
	}
	
	public double getAlpha() {
		return m_alpha;
	}

	private void findAlpha(Instances data) throws Exception {
		double minAlpha = 0, minError = Double.POSITIVE_INFINITY, currErr;

		for (int i = -17; i < 0; i++) {
			m_alpha = Math.pow(3, i);
			currErr = gradientDescentForFindingAlpha(ITERATION_LIMIT, data);

			if (currErr < minError) {
				minError = currErr;
				minAlpha = m_alpha;
			}
		}

		m_alpha = minAlpha;
	}

	private double gradientDescentForFindingAlpha(int limit, Instances trainingData) throws Exception {
		int index = 0;
		double prevErr = Double.POSITIVE_INFINITY, currErr;
		double[] coefficients = new double[m_truNumAttributes + 1];
		double[] tempCoefficients = new double[m_truNumAttributes + 1];

		Arrays.fill(coefficients, 1);

		while (limit > 0) {
			index++;
			limit--;

			updateCoefficients(coefficients, trainingData, tempCoefficients);

			// Checking the error every 100 loops
			if (index % 100 != 0) {
				continue;
			}

			currErr = calculateMSE(coefficients, trainingData);

			if (currErr > prevErr) {
				break;
			} else {
				prevErr = currErr;
			}
		}

		return prevErr;
	}

	/**
	 * An implementation of the gradient descent algorithm which should return the
	 * weights of a linear regression predictor which minimizes the average squared
	 * error.
	 * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		int index = 0;
		double prevErr = Double.POSITIVE_INFINITY, currErr;
		double[] coefficients = new double[m_truNumAttributes + 1];
		double[] tempCoefficients = new double[m_truNumAttributes + 1];

		Arrays.fill(coefficients, 1);

		while (true) {
			index++;
			
			updateCoefficients(coefficients, trainingData, tempCoefficients);

			// Checking the error every 100 loops
			if (index % 100 != 0) {
				continue;
			}

			currErr = calculateMSE(coefficients, trainingData);

			if (Math.abs(currErr - prevErr) < STOPPING_GD) {
				break;
			} else {
				prevErr = currErr;
			}
		}

		return coefficients;
	}

	private void updateCoefficients(double[] coefficients, Instances trainingData, double[] tempCoefficients) {
		// Calculate the new tetas.
		for (int i = 0; i < coefficients.length; i++) {
			tempCoefficients[i] = calcNewCoefficient(i, coefficients, trainingData);
		}

		// Coping the temporary values into the real values.
		for (int i = 0; i < tempCoefficients.length; i++) {
			coefficients[i] = tempCoefficients[i];
		}
	}

	private double calcNewCoefficient(int index, double[] coefficients, Instances trainingData) {
		double tempInstanceSum = 0, productForInnerProduct = 1;

		// Calculating the sum of the instances
		for (Instance instance : trainingData) {
			// if we are not on teta 0.
			if (index != 0) {
				productForInnerProduct = instance.value(index - 1);
			}

			tempInstanceSum += (innerProduct(coefficients, instance) - instance.value(m_ClassIndex))
					* productForInnerProduct;
		}

		return coefficients[index] - (m_alpha * tempInstanceSum / trainingData.numInstances());
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights given by
	 * m_coefficients on a single instance.
	 *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		return innerProduct(m_coefficients, instance);
	}

	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
	 *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances trainingData) throws Exception {
		return calculateMSE(m_coefficients, trainingData);
	}

	private double calculateMSE(double[] coefficients, Instances trainingData) throws Exception {
		double tempInstanceSum = 0;

		// Calculating the sum of the instances
		for (Instance instance : trainingData) {
			tempInstanceSum += Math.pow(innerProduct(coefficients, instance) - instance.value(m_ClassIndex), 2);
		}

		return tempInstanceSum / (trainingData.numInstances() * 2);
	}

	private double innerProduct(double[] coefficients, Instance instance) {
		double innerProduct = coefficients[0];

		for (int i = 0; i < m_truNumAttributes; i++) {
			if (i == m_ClassIndex) {
				continue;
			}

			innerProduct += coefficients[i + 1] * instance.value(i);
		}

		return innerProduct;
	}

	@Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
