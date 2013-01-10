package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Map;
import java.util.Scanner;

public class LogisticRegressionWithHashing {
	public class Weights {
		int w0;
		double[] ws;
		int dim;

		public Weights(int dim) {
			w0 = 0;
			this.dim = dim;
			ws = new double[dim];
		}

		public double l2norm() {
			double l2 = w0 * w0;
			for (double w : ws)
				l2 += w * w;
			return Math.sqrt(l2);
		}
	}

	private double computeWeightFeatureProduct(Weights weights,
			Map<Integer, Integer> hashedfeature) {
		double prod = weights.w0;
		for (Map.Entry<Integer, Integer> entry: hashedfeature.entrySet()) {
			prod += weights.ws[entry.getKey()] * entry.getValue();
		}
		return prod;
	}

	private void updateWeights(Weights weights, Map<Integer, Integer> hashedfeature, double step,
			double grad, double lambda) {
		weights.w0 += -step * grad;
		// update weights along the negative gradient
		for (Map.Entry<Integer, Integer> entry: hashedfeature.entrySet()) {
			int key = entry.getKey();
			weights.ws[key] += -step * (grad * entry.getValue() + lambda * weights.ws[key]);
		}
	}

	public Weights train(DataSet dataset, int dim, double lambda, double step,
			boolean personalized) {
		Weights weights = new Weights(dim);
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (dataset.hasNext()) {
			HashedDataInstance instance = dataset.nextHashedInstance(dim,
					personalized);
			// compute w0 + <w, x>
			double wx = computeWeightFeatureProduct(weights,
					instance.hashedFeature);
			double exp = Math.exp(wx);
			exp = Double.isInfinite(exp) ? (Double.MAX_VALUE - 1) : exp;
			int num_positive = instance.clicks;
			int num_negative = instance.impressions - num_positive;
			// compute the gradient
			double grad = num_positive * (-1 / (1 + exp)) + (num_negative)
					* (exp / (1 + exp));
			updateWeights(weights, instance.hashedFeature, step, grad, lambda);
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
				System.err.println("l2 norm of weights: " + weights.l2norm());
			}
		}
		if (count < dataset.size) {
			System.err
					.println("Warning: the real size of the data is less than the input size: "
							+ dataset.size + "<" + count);
		}
		System.err.println("Done. Total processed instances: " + count);
		dataset.reset();
		return weights;
	}

	public ArrayList<Double> predict(Weights weights, DataSet dataset,
			boolean personalized) {
		ArrayList<Double> ctr = new ArrayList<Double>();
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (dataset.hasNext()) {
			HashedDataInstance instance = dataset.nextHashedInstance(
					weights.dim, personalized);
			double wx = computeWeightFeatureProduct(weights,
					instance.hashedFeature);
			double exp = Math.exp(wx);
			if (Double.isInfinite(exp))
				exp = Double.MAX_VALUE-1;
			ctr.add(exp / (1 + exp));
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}
		if (count < dataset.size) {
			System.err
					.println("Warning: the real size of the data is less than the input size: "
							+ dataset.size + "<" + count);
		}
		System.err.println("Done. Total processed instances: " + count);
		dataset.reset();
		return ctr;
	}

	public double eval(String pathToSol, ArrayList<Double> ctr_prediction) {
		try {
			Scanner sc = new Scanner(new BufferedReader(new FileReader(
					pathToSol)));
			int size = ctr_prediction.size();
			double wmse = 0.0;
			int total = 0;
			for (int i = 0; i < size; i++) {
				String[] fields = sc.nextLine().split(",");
				int clicks = Integer.parseInt(fields[0]);
				int impressions = Integer.parseInt(fields[1]);
				total += impressions;
				double ctr = (double) clicks / impressions;
				wmse += Math.pow((ctr - ctr_prediction.get(i)), 2);
			}
			wmse /= total;
			return Math.sqrt(wmse);
		} catch (Exception e) {
			e.printStackTrace();
			return Double.MAX_VALUE;
		}
	}

	public static void main(String args[]) throws IOException {
		boolean personal = true;
		int training_size = personal ? DataSet.TRAININGSIZE
				: DataSet.TESTINGSIZE;
		training_size = (int) (DataSet.TESTINGSIZE * 5);
		int testing_size = DataSet.TESTINGSIZE;
		DataSet training = new DataSet(
				"/Users/haijieg/workspace/kdd2012/datawithfeature/train.txt",
				true, training_size);
		DataSet testing = new DataSet(
				"/Users/haijieg/workspace/kdd2012/datawithfeature/test.txt",
				false, testing_size);

		DecimalFormat formatter = new DecimalFormat("###.##");
		LogisticRegressionWithHashing lr = new LogisticRegressionWithHashing();
		double step = 0.01;
		double lambda = 0;
		int[] dims = {1572869};
		for (int dim : dims) {
			System.err.println("Running dim = " + dim);
			String ofname = "/Users/haijieg/workspace/kdd2012/experiments/hashing";
			if (personal)
				ofname += "_personal";
			ofname += "/ctr_" + dim + ".txt";
			Weights weights = lr.train(training, dim, lambda, step, personal);
			ArrayList<Double> ctr_prediction = lr.predict(weights, testing,
					personal);
			BufferedWriter out = new BufferedWriter(new FileWriter(ofname));
			String solpath = "/Users/haijieg/workspace/kdd2012/solution/sol.txt";
			double wmse = lr.eval(solpath, ctr_prediction);
			out.write("step: " + step + "\n");
			out.write("hash dim: " + dim + "\n");
			out.write("wmse: " + +wmse + "\n");
			System.out.println("wmse: " + wmse + "\n");
			for (double ctr : ctr_prediction) {
				out.write(formatter.format(ctr) + "\n");
			}
			out.close();
		}
	}
}