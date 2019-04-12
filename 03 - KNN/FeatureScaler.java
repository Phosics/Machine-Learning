package HomeWork3;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	private Standardize standardizeFilter = new Standardize();

	/**
	 * Returns a scaled version (using standarized normalization) of the given
	 * dataset.
	 * 
	 * @param instances
	 *            The original dataset.
	 * @return A scaled instances object.
	 * @throws Exception
	 */
	public Instances scaleData(Instances instances) throws Exception {
		standardizeFilter.setInputFormat(instances);

		return Filter.useFilter(instances, standardizeFilter);
	}
}