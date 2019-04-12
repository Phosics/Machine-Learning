package HomeWork2;

import java.util.HashSet;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
	double splittedValue;

	boolean isLeaf() {
		return children.length == 0;
	}
}

public class DecisionTree implements Classifier {
	private static final int RECURRENCE_EVENTS_INDEX = 0;
	private static final int NO_RECURRENCE_EVENTS_INDEX = 1;

	private static final double[][] CHI_SQUARE_TABLE = { { 0, 0.102, 0.455, 1.323, 3.841, 7.879 },
			{ 0, 0.575, 1.386, 2.773, 5.991, 10.597 }, { 0, 1.213, 2.366, 4.108, 7.815, 12.838 },
			{ 0, 1.923, 3.357, 5.385, 9.488, 14.860 }, { 0, 2.675, 4.351, 6.626, 11.070, 16.750 },
			{ 0, 3.455, 5.348, 7.841, 12.592, 18.548 }, { 0, 4.255, 6.346, 9.037, 14.067, 20.278 },
			{ 0, 5.071, 7.344, 10.219, 15.507, 21.955 }, { 0, 5.899, 8.343, 11.389, 16.919, 23.589 },
			{ 0, 6.737, 9.342, 12.549, 18.307, 25.188 }, { 0, 7.584, 10.341, 13.701, 19.675, 26.757 },
			{ 0, 8.438, 11.340, 14.845, 21.026, 28.300 } };

	private Node rootNode;
	private boolean isGini;
	private int pValueIndex;
	private int maxTreeHeight;
	private double avgTreeHeight;

	public DecisionTree(boolean isGini, int pValueIndex) {
		this.isGini = isGini;
		this.pValueIndex = pValueIndex;
	}

	/**
	 * Get the max level an instance got on the tree.
	 * 
	 * @return
	 */
	public int getMaxTreeHeight() {
		return maxTreeHeight;
	}

	/**
	 * Get the average level instances got on the tree.
	 * 
	 * @return
	 */
	public double getAvgTreeHeight() {
		return avgTreeHeight;
	}

	/**
	 * Get the p_value of the chi square test.
	 * 
	 * @return
	 */
	public int getPValueIndex() {
		return pValueIndex;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		rootNode = buildTree(arg0);
	}

	@Override
	public double classifyInstance(Instance instance) {
		Node currNode = rootNode, child = rootNode;
		int height = 0;

		while (!currNode.isLeaf() && child == currNode) {
			for (int i = 0; i < currNode.children.length; i++) {
				child = currNode.children[i];

				// If the current child does not exists or the values not match
				if (child == null || child.splittedValue != instance.value(currNode.attributeIndex)) {
					continue;
				}

				currNode = child;
				height++;
				break;
			}
		}

		updateHeight(height);

		return currNode.returnValue;
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

	public void printTree() {
		StringBuilder builder = new StringBuilder();

		builder.append("Root\n");
		printReturningValue(builder, rootNode, 0);
		printChilds(builder, rootNode, 1);

		System.out.println(builder.toString());
	}
	
	public double calcAvgError(Instances data) {
		int mistakes = 0;

		maxTreeHeight = 0;
		avgTreeHeight = 0;

		for (Instance instance : data) {
			if (classifyInstance(instance) != instance.classValue()) {
				mistakes++;
			}
		}

		avgTreeHeight /= data.size();

		return (double) mistakes / data.numInstances();
	}
	
	private Node buildTree(Instances nodeData) {
		double maxGain = 0, gain;
		int maxAttributeIndex = -1;
		Node child, currNode = new Node();

		for (int attributeIndex = 0; attributeIndex < nodeData.numAttributes() - 1; attributeIndex++) {
			gain = calcGain(nodeData, attributeIndex);

			if (gain > maxGain) {
				maxGain = gain;
				maxAttributeIndex = attributeIndex;
			}
		}

		currNode.returnValue = getMaxClassValue(nodeData);

		// If the attribute will not change the tree OR Chi Square Test
		if (maxGain == 0 || isPrunningNeeded(nodeData, maxAttributeIndex)) {
			currNode.children = new Node[0];

			return currNode;
		}

		Attribute maxAttribute = nodeData.attribute(maxAttributeIndex);
		currNode.attributeIndex = maxAttributeIndex;
		Node[] children = new Node[maxAttribute.numValues()];

		// Creating the children of the current node.
		for (int i = 0; i < maxAttribute.numValues(); i++) {
			Instances childData = filterDataByAttributeValue(nodeData, maxAttributeIndex, maxAttribute.value(i));

			// If no data for the child
			if (childData.size() == 0) {
				continue;
			}

			child = buildTree(childData);

			children[i] = child;
			child.parent = currNode;
			child.splittedValue = childData.instance(0).value(maxAttributeIndex);
		}

		currNode.children = children;

		return currNode;
	}

	private double calcGain(Instances nodeData, int attributeIndex) {
		Attribute attribute = nodeData.attribute(attributeIndex);
		double currNodeImpurity, childImpurity, goodnessOfSplit = 0;

		if (isGini) {
			currNodeImpurity = calcGini(createProbabilities(nodeData));
		} else {
			currNodeImpurity = calcEntropy(createProbabilities(nodeData));
		}

		for (int i = 0; i < attribute.numValues(); i++) {
			Instances childData = filterDataByAttributeValue(nodeData, attributeIndex, attribute.value(i));

			// If there are no instances after the filter the child wont be created
			if (childData.size() == 0) {
				continue;
			}

			double[] childProbabilities = createProbabilities(childData);

			if (isGini) {
				childImpurity = calcGini(childProbabilities);
			} else {
				childImpurity = calcEntropy(childProbabilities);
			}

			goodnessOfSplit += childData.numInstances() * childImpurity / nodeData.numInstances();
		}

		return currNodeImpurity - goodnessOfSplit;
	}

	private double calcEntropy(double[] probabilities) {
		double entropyIndex = 0;

		for (double probabilty : probabilities) {

			// If the probability = 0, we have log(0), so we avoid this.
			if (probabilty == 0) {
				continue;
			}

			entropyIndex += probabilty * log2(probabilty);
		}

		return -entropyIndex;
	}

	private double calcGini(double[] probabilities) {
		double giniIndex = 0;

		for (double probabilty : probabilities) {
			giniIndex += Math.pow(probabilty, 2);
		}

		return 1 - giniIndex;
	}

	private double calcChiSquare(Instances nodeData, int attributeIndex) {
		Attribute attribute = nodeData.attribute(attributeIndex);
		Instances childData;
		double e0, e1, summary = 0;

		double[] probabilities = createProbabilities(nodeData), childClassValuesCounters;

		for (int i = 0; i < attribute.numValues(); i++) {
			childData = filterDataByAttributeValue(nodeData, attributeIndex, attribute.value(i));
			childClassValuesCounters = countClassValues(childData);

			e0 = childData.size() * probabilities[RECURRENCE_EVENTS_INDEX];
			e1 = childData.size() * probabilities[NO_RECURRENCE_EVENTS_INDEX];

			// If e0 = 0 we divide by 0, so if e0 = 0 we add 0 to the summary
			if (e0 != 0) {
				summary += Math.pow(childClassValuesCounters[RECURRENCE_EVENTS_INDEX] - e0, 2) / e0;
			}

			// If e1 = 0 we divide by 0, so if e1 = 0 we add 0 to the summary
			if (e1 != 0) {
				summary += Math.pow(childClassValuesCounters[NO_RECURRENCE_EVENTS_INDEX] - e1, 2) / e1;
			}
		}

		return summary;
	}

	private boolean isPrunningNeeded(Instances data, int attributeIndex) {
		return CHI_SQUARE_TABLE[calcDegreeOfFreedom(data, attributeIndex)][pValueIndex] > calcChiSquare(data,
				attributeIndex);
	}

	private int calcDegreeOfFreedom(Instances data, int attributeIndex) {
		Set<Double> values = new HashSet<>();

		for (Instance instance : data) {
			values.add(instance.value(attributeIndex));
		}

		return values.size() - 2;
	}

	private Instances filterDataByAttributeValue(Instances nodeData, int attributeIndex, String attributeValue) {
		Instances sonData = new Instances(nodeData);
		Instance instance;

		for (int i = nodeData.numInstances() - 1; i > -1; i--) {
			instance = nodeData.get(i);

			// Removing instances without the given attribute value
			if (!instance.stringValue(attributeIndex).equals(attributeValue)) {
				sonData.remove(i);
			}
		}

		return sonData;
	}

	private double[] createProbabilities(Instances nodeData) {
		double[] probabilities = countClassValues(nodeData);

		// Dividing each counter by the number of attributes.
		for (int i = 0; i < probabilities.length; i++) {
			probabilities[i] /= nodeData.numInstances();
		}

		return probabilities;
	}

	private double getMaxClassValue(Instances nodeData) {
		double maxCounter = -1, maxValue = -1;
		double[] classValueCounters = countClassValues(nodeData);

		for (int i = 0; i < classValueCounters.length; i++) {
			if (classValueCounters[i] > maxCounter) {
				maxCounter = classValueCounters[i];
				maxValue = (double) i;
			}
		}

		return maxValue;
	}

	private void updateHeight(int height) {
		if (maxTreeHeight < height) {
			maxTreeHeight = height;
		}

		avgTreeHeight += height;
	}
	
	private double[] countClassValues(Instances data) {
		double[] yValueCounters = new double[data.numClasses()];

		for (Instance instance : data) {
			yValueCounters[(int) instance.classValue()]++;
		}

		return yValueCounters;
	}

	private double log2(double number) {
		return Math.log(number) / Math.log(2);
	}

	private void printTree(StringBuilder builder, Node node, int level) {
		printLevel(builder, level);

		builder.append(String.format("If attribute %s = %s\n", node.parent.attributeIndex, node.splittedValue));
		printReturningValue(builder, node, level);
		printChilds(builder, node, level + 1);
	}

	private void printChilds(StringBuilder builder, Node node, int level) {
		for (Node child : node.children) {
			if (child != null) {
				printTree(builder, child, level);
			}
		}
	}

	private void printReturningValue(StringBuilder builder, Node node, int level) {
		printLevel(builder, level);

		if (node.isLeaf()) {
			builder.append("\tLeaf. ");
		}

		builder.append(String.format("Returning value:%s\n", node.returnValue));
	}

	private void printLevel(StringBuilder builder, int level) {
		if (level != 0) {
			builder.append(String.format("%" + level + "s", "").replace(" ", "\t"));
		}
	}
}
