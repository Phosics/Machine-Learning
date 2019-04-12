package HomeWork3;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

import HomeWork3.Knn.DistanceCheck;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * A helper class for the priority queue to find the k nearest neighbors.
 * 
 * @author itzik
 *
 */
class Pair implements Comparable<Pair> {
	private Instance instance;
	private double distance;

	public Pair(Instance instance, double distance) {
		this.instance = instance;
		this.distance = distance;
	}

	public Instance getInstance() {
		return this.instance;
	}

	public double getDistance() {
		return this.distance;
	}

	@Override
	public int compareTo(Pair o) {
		return Double.compare(o.getDistance(), distance);
	}
}

class DistanceCalculator {
	private static final int NO_EFFICIENT_DISTANCE_CHECK = -1;

	private double lp;

	public void setLp(double lp) {
		this.lp = lp;
	}

	/**
	 * We leave it up to you wheter you want the distance method to get all relevant
	 * parameters(lp, efficient, etc..) or have it has a class variables.
	 */
	public double distance(Instance one, Instance two, DistanceCheck distanceCheck, double kthNeighborDistance) {
		return (distanceCheck == DistanceCheck.Efficient && kthNeighborDistance > NO_EFFICIENT_DISTANCE_CHECK)
				? efficientDistance(one, two, kthNeighborDistance)
				: normalDistance(one, two);
	}

	/**
	 * For when doing regular distance check.
	 */
	public double distance(Instance one, Instance two) {
		return distance(one, two, DistanceCheck.Regular, NO_EFFICIENT_DISTANCE_CHECK);
	}

	private double efficientDistance(Instance one, Instance two, double kthNeighborDistance) {
		return lp == Double.POSITIVE_INFINITY ? efficientLInfinityDistance(one, two, kthNeighborDistance)
				: efficientLpDisatnce(one, two, Math.pow(kthNeighborDistance, lp));
	}

	private double normalDistance(Instance one, Instance two) {
		return lp == Double.POSITIVE_INFINITY ? lInfinityDistance(one, two) : lpDisatnce(one, two);
	}

	/**
	 * Returns the Lp distance between 2 instances.
	 * 
	 * @param one
	 * @param two
	 */
	private double lpDisatnce(Instance one, Instance two) {
		double distance = 0;

		for (int i = 0; i < one.numAttributes() - 1; i++) {
			distance += Math.pow(Math.abs(one.value(i) - two.value(i)), lp);
		}

		return Math.pow(distance, 1 / lp);
	}

	/**
	 * Returns the L infinity distance between 2 instances.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private double lInfinityDistance(Instance one, Instance two) {
		double distance = 0, maxDistance = 0;

		for (int i = 0; i < one.numAttributes() - 1; i++) {
			distance = Math.abs(one.value(i) - two.value(i));

			if (distance > maxDistance) {
				maxDistance = distance;
			}
		}

		return maxDistance;
	}

	/**
	 * Returns the Lp distance between 2 instances, while using an efficient
	 * distance check.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private double efficientLpDisatnce(Instance one, Instance two, double kthNeighborDistance) {
		double distance = 0;

		for (int i = 0; i < one.numAttributes() - 1; i++) {
			distance += Math.pow(Math.abs(one.value(i) - two.value(i)), lp);

			// If the current neighbor is more far then the Kth neighbor
			if (kthNeighborDistance < distance) {
				return distance;
			}
		}

		return distance;
	}

	/**
	 * Returns the Lp distance between 2 instances, while using an efficient
	 * distance check.
	 * 
	 * @param one
	 * @param two
	 * @return
	 */
	private double efficientLInfinityDistance(Instance one, Instance two, double kthNeighborDistance) {
		double distance = 0, maxDistance = 0;

		for (int i = 0; i < one.numAttributes() - 1; i++) {
			distance = Math.abs(one.value(i) - two.value(i));

			if (distance > maxDistance) {
				maxDistance = distance;
			}

			// If the current neighbor is more far then the Kth neighbor
			if (kthNeighborDistance < maxDistance) {
				return maxDistance;
			}
		}

		return maxDistance;
	}
}

public class Knn implements Classifier {

	public enum DistanceCheck {
		Regular("regular"), Efficient("efficient");

		private String stringValue;

		private DistanceCheck(String stringValue) {
			this.stringValue = stringValue;
		}

		public String getStringValue() {
			return this.stringValue;
		}
	}

	public enum Weighted {
		WEIGHTED("weighted"), UNIFORM("uniform");

		private String stringValue;

		private Weighted(String stringValue) {
			this.stringValue = stringValue;
		}

		public String getStringValue() {
			return this.stringValue;
		}
	}

	private DistanceCalculator distanceCalculator = new DistanceCalculator();
	private int k;
	private Weighted weightedDistance;
	private Instances m_trainingInstances;
	private DistanceCheck distanceCheck;

	private int totalTimeElapse;
	private double averageTimeElapse;

	public int getTotalTimeElapse() {
		return totalTimeElapse;
	}

	public double getAverageTimeElapse() {
		return averageTimeElapse;
	}

	public void setK(int k) {
		this.k = k;
	}

	public void setWeightedDistance(Weighted weightedDistance) {
		this.weightedDistance = weightedDistance;
	}

	public void setLP(double lp) {
		distanceCalculator.setLp(lp);
	}

	public void setDistanceCheck(DistanceCheck distanceCheck) {
		this.distanceCheck = distanceCheck;
	}

	@Override
	/**
	 * Build the knn classifier. In our case, simply stores the given instances for
	 * later use in the prediction.
	 * 
	 * @param instances
	 */
	public void buildClassifier(Instances instances) throws Exception {
		this.m_trainingInstances = instances;
	}

	/**
	 * Returns the knn prediction on the given instance.
	 * 
	 * @param instance
	 * @return The instance predicted value.
	 */
	public double regressionPrediction(Instance instance) {
		PriorityQueue<Pair> neighbors = findNearestNeighbors(instance);

		if (weightedDistance == Weighted.WEIGHTED) {
			return getWeightedAverageValue(neighbors);
		}

		return getAverageValue(neighbors);
	}

	/**
	 * Caclcualtes the average error on a give set of instances. The average error
	 * is the average absolute error between the target value and the predicted
	 * value across all insatnces.
	 * 
	 * @param insatnces
	 * @return
	 */
	public double calcAvgError(Instances insatnces) {
		double errorSum = 0;

		for (Instance instance : insatnces) {
			errorSum += Math.abs(regressionPrediction(instance) - instance.classValue());
		}

		return errorSum / insatnces.numInstances();
	}

	/**
	 * Calculates the cross validation error, the average error on all folds.
	 * 
	 * @param insances
	 *            Insances used for the cross validation
	 * @param num_of_folds
	 *            The number of folds to use.
	 * @return The cross validation error.
	 * @throws Exception
	 */
	public double crossValidationError(Instances insances, int num_of_folds) throws Exception {
		List<Instances> arrayOfListsOfInstances = createListOfInstances(insances, num_of_folds);
		totalTimeElapse = 0;

		// Adding the instances to the list.
		for (int i = 0; i < insances.numInstances(); i++) {
			arrayOfListsOfInstances.get(i % num_of_folds).add(insances.get(i));
		}

		double sumErrors = 0, time;

		// Calculating the errors
		for (int validationIndex = 0; validationIndex < num_of_folds; validationIndex++) {
			buildClassifier(joinTrainingGroups(arrayOfListsOfInstances, validationIndex, num_of_folds));

			time = System.nanoTime();
			sumErrors += calcAvgError(arrayOfListsOfInstances.get(validationIndex));
			totalTimeElapse += System.nanoTime() - time;
		}

		averageTimeElapse = totalTimeElapse / (double) num_of_folds;

		return sumErrors / num_of_folds;
	}

	/**
	 * Finds the k nearest neighbors.
	 * 
	 * @param instance
	 */
	public PriorityQueue<Pair> findNearestNeighbors(Instance instance) {
		PriorityQueue<Pair> maxHeap = new PriorityQueue<>(k);
		Instance currInstance;
		double currDistance;

		// Adding the first k instances to the heap
		for (int i = 0; i < k; i++) {
			currInstance = m_trainingInstances.get(i);

			maxHeap.add(new Pair(currInstance, distanceCalculator.distance(instance, currInstance)));
		}

		// Calculate all the distances
		for (int i = k; i < m_trainingInstances.numInstances(); i++) {
			currInstance = m_trainingInstances.get(i);
			currDistance = distanceCalculator.distance(instance, currInstance, distanceCheck,
					maxHeap.peek().getDistance());

			// If the current Instance is closer then the kth neighbor
			if (maxHeap.peek().getDistance() > currDistance) {
				maxHeap.poll();
				maxHeap.add(new Pair(currInstance, currDistance));
			}
		}

		return maxHeap;
	}

	/**
	 * Cacluates the average value of the given elements in the collection.
	 * 
	 * @param
	 * @return
	 */
	public double getAverageValue(PriorityQueue<Pair> neighbors) {
		double sum = 0;

		for (Pair neighbor : neighbors) {
			// If the current neighbor is just like the instance
			if (neighbor.getDistance() == 0) {
				return neighbor.getInstance().classValue();
			}

			sum += neighbor.getInstance().classValue();
		}

		return sum / neighbors.size();
	}

	/**
	 * Calculates the weighted average of the target values of all the elements in
	 * the collection with respect to their distance from a specific instance.
	 * 
	 * @return
	 */
	public double getWeightedAverageValue(PriorityQueue<Pair> neighbors) {
		double sum = 0, weightedAverage = 0, weight, distance;

		for (Pair neighbor : neighbors) {
			distance = neighbor.getDistance();

			// If the current neighbor is just like the instance
			if (distance == 0) {
				return neighbor.getInstance().classValue();
			}

			weight = 1 / Math.pow(distance, 2);

			sum += weight * neighbor.getInstance().classValue();
			weightedAverage += weight;
		}

		return sum / weightedAverage;
	}
	
	private List<Instances> createListOfInstances(Instances instances, int numOfFolds) {
		List<Instances> listsOfInstances = new ArrayList<Instances>(numOfFolds);

		for (int i = 0; i < numOfFolds; i++) {
			listsOfInstances.add(new Instances(instances, (instances.numInstances() / numOfFolds) + 1));
		}

		return listsOfInstances;
	}
	
	private Instances joinTrainingGroups(List<Instances> arrayOfListsOfInstances, int indexOfValidation, int numOfFolds) {
		Instances trainingDataset = new Instances(arrayOfListsOfInstances.get(0), numOfFolds);

		for (int i = 0; i < numOfFolds; i++) {
			// If the current group is the validation group
			if (i == indexOfValidation) {
				continue;
			}

			trainingDataset.addAll(arrayOfListsOfInstances.get(i));
		}

		return trainingDataset;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub - You can ignore.
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub - You can ignore.
		return null;
	}

	@Override
	public double classifyInstance(Instance instance) {
		// TODO Auto-generated method stub - You can ignore.
		return 0.0;
	}
}
