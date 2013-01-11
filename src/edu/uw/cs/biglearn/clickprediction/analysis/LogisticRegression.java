package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class LogisticRegression {
	/**
	 * This class represents the weights in the logistic regression model.
	 * 
	 * @author haijieg
	 * 
	 */
	public class Weights {
		double w0;
		/*
		 * query.get("123") will return the weight for the feature:
		 * "token 123 in the query field".
		 */
		Map<Integer, Double> query;
		Map<Integer, Double> title;
		Map<Integer, Double> keyword;
		Map<Integer, Double> description;
		double wPosition;
		double wDepth;
		double wAge;
		double wGender;
		
		Map<Integer, Integer> accessTime_query; // keep track of the access timestamp of feature weights.
																		 // Using this to do delayed regularization.
		Map<Integer, Integer> accessTime_title; 	
		Map<Integer, Integer> accessTime_description;
		Map<Integer, Integer> accessTime_keyword;	


		public Weights() {
			w0 = wAge = wGender = wDepth = wPosition = 0.0;
			query = new HashMap<Integer, Double>();
			title = new HashMap<Integer, Double>();
			keyword = new HashMap<Integer, Double>();
			description = new HashMap<Integer, Double>();
			accessTime_query = new HashMap<Integer, Integer>();
			accessTime_keyword = new HashMap<Integer, Integer>();
			accessTime_title = new HashMap<Integer, Integer>();
			accessTime_description = new HashMap<Integer, Integer>();
		}

		@Override
		public String toString() {
			DecimalFormat myFormatter = new DecimalFormat("###.##");
			StringBuilder builder = new StringBuilder();
			builder.append("Intercept: " + myFormatter.format(w0) + "\n");
			builder.append("Depth: " + myFormatter.format(wDepth) + "\n");
			builder.append("Position: " + myFormatter.format(wPosition) + "\n");
			builder.append("Gender: " + myFormatter.format(wGender) + "\n");
			builder.append("Age: " + myFormatter.format(wAge) + "\n");
			builder.append("query: " + query.toString() + "\n");
			builder.append("title: " + title.toString() + "\n");
			builder.append("keyword: " + keyword.toString() + "\n");
			builder.append("description: " + description.toString() + "\n");
			return builder.toString();
		}

		/**
		 * @return the l2 norm of this weight vector.
		 */
		public double l2norm() {
			double l2 = w0 * w0;
			for (double w : query.values())
				l2 += w * w;
			for (double w : title.values())
				l2 += w * w;
			for (double w : description.values())
				l2 += w * w;
			for (double w : keyword.values())
				l2 += w * w;
			return Math.sqrt(l2);
		}

		/**
		 * @return the l0 norm of this weight vector.
		 */
		public int l0norm() {
			return 3 + query.size() + title.size() + keyword.size()
					+ description.size();
		}
	}
	


	/**
	 * Compute w^Tx.
	 * 
	 * @param weights
	 * @param instance
	 * @return
	 */
	private double computeWeightFeatureProduct(Weights weights,
			DataInstance instance) {
		double wx = weights.w0 + weights.wAge * instance.age + weights.wGender
				* instance.gender + weights.wDepth * instance.depth
				+ weights.wPosition * instance.position;
		for (int token : instance.query) {
			Double w_token = weights.query.get(token);
			if (w_token != null)
				wx += w_token;
		}
		for (int token : instance.title) {
			Double w_token = weights.title.get(token);
			if (w_token != null)
				wx += w_token;
		}
		for (int token : instance.keyword) {
			Double w_token = weights.keyword.get(token);
			if (w_token != null)
				wx += w_token;
		}
		for (int token : instance.description) {
			Double w_token = weights.description.get(token);
			if (w_token != null)
				wx += w_token;
		}
		return wx;
	}

	/**
	 * Update the weights based on the current instance.
	 * 
	 * @param weights
	 * @param instance
	 * @param step
	 * @param lambda
	 *            the Regularization parameter
	 */
	private void updateWeights(Weights weights, DataInstance instance,
			double step, double lambda, int timestamp) {

		// Perform delayed regularization
		if (lambda > 1e-8) {
  		performDelayedRegularization(instance.query, weights.query,
  				weights.accessTime_query, timestamp, step, lambda);
  		performDelayedRegularization(instance.title, weights.title,
  				weights.accessTime_title, timestamp, step, lambda);
  		performDelayedRegularization(instance.keyword, weights.keyword,
  				weights.accessTime_keyword, timestamp, step, lambda);
  		performDelayedRegularization(instance.description, weights.description,
  				weights.accessTime_description, timestamp, step, lambda);
		}

		
		// compute w0 + <w, x>
		double wx = computeWeightFeatureProduct(weights, instance);
		double exp = Math.exp(wx);
		exp = Double.isInfinite(exp) ? (Double.MAX_VALUE - 1) : exp;
		int num_positive = instance.clicks;
		int num_negative = instance.impressions - num_positive;
		// compute the gradient
		double grad = num_positive * (-1 / (1 + exp)) + (num_negative)
				* (exp / (1 + exp));

		// update weights along the negative gradient
		weights.w0 += -step * grad;
		weights.wAge += -step * (grad * instance.age + lambda * weights.wAge);
		weights.wGender += -step
				* (grad * instance.gender + lambda * weights.wGender);
		weights.wDepth += -step
				* (grad * instance.depth + lambda * weights.wDepth);
		weights.wPosition += -step
				* (grad * instance.position + lambda * weights.wPosition);
		for (int token : instance.query) {
			Double w = weights.query.get(token);
			if (w == null)
				w = 0.0;
			w += -step * (grad + lambda * w);
			weights.query.put(token, w);
		}
		for (int token : instance.title) {
			Double w = weights.title.get(token);
			if (w == null)
				w = 0.0;
			w += -step * (grad + lambda * w);
			weights.title.put(token, w);
		}
		for (int token : instance.keyword) {
			Double w = weights.keyword.get(token);
			if (w == null)
				w = 0.0;
			w += -step * (grad + lambda * w);
			weights.keyword.put(token, w);
		}
		for (int token : instance.description) {
			Double w = weights.description.get(token);
			if (w == null)
				w = 0.0;
			w += -step * (grad + lambda * w);
			weights.description.put(token, w);
		}
	}

	
	/**
	 * Apply delayed regularization to the weights corresponding to the given tokens.
	 * @param tokens
	 * @param weights
	 * @param accessTime	a lookup table for querying the last access timestamp of a given weight.
	 * @param now 	the current timestamp.
	 * @param step
	 * @param lambda
	 */
	private void performDelayedRegularization(int[] tokens,
			Map<Integer, Double> weights, Map<Integer, Integer> accessTime,
			int now, double step, double lambda) {
		for (int token : tokens) {
			Integer t = accessTime.get(token);
			if (t != null) {
				double w = weights.get(token);
				weights.put(token, w * Math.pow((1 - step * lambda), now-t-1));
			}
			accessTime.put(token, now);
		}
	}
	
	/**
	 * Train the logistic regression model using the training data and the
	 * hyperparameters.
	 * 
	 * @param dataset
	 * @param lambda
	 * @param step
	 * @return the weights for the model.
	 */
	public Weights train(DataSet dataset, double lambda, double step) {
		Weights weights = new Weights();
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		for (int i = 0; i < 1; i++) {
  		while (dataset.hasNext()) {
  			DataInstance instance = dataset.nextInstance();
  
  			updateWeights(weights, instance, step, lambda, count);
  			count++;
  			if (count % 100000 == 0) {
  				System.err.println("Processed " + count + " lines");
  				System.err.println("l2 norm of weights: " + weights.l2norm());
  				System.err.println("l0 norm of weights: " + weights.l0norm());
  			}
  		}
  		if (count < dataset.size) {
  			System.err
  					.println("Warning: the real size of the data is less than the input size: "
  							+ dataset.size + "<" + count);
  		}
  		System.err.println("Done. Total processed instances: " + count);
  		dataset.reset();
		}
		return weights;
	}

	/**
	 * Using the weights to predict CTR in for the test dataset.
	 * 
	 * @param weights
	 * @param dataset
	 * @return An array storing the CTR for each datapoint in the test data.
	 */
	public ArrayList<Double> predict(Weights weights, DataSet dataset) {
		ArrayList<Double> ctr = new ArrayList<Double>();
		System.err.println("Loading data from " + dataset.path + " ... ");
		int count = 0;
		while (dataset.hasNext()) {
			DataInstance instance = dataset.nextInstance();
			double wx = computeWeightFeatureProduct(weights, instance);
			double exp = Math.exp(wx);
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

	/**
	 * Evaluates the model by computing the weighted rooted mean square error of
	 * between the prediction and the true labels.
	 * 
	 * @param pathToSol
	 * @param ctr_prediction
	 * @return the weighted rooted mean square error of between the prediction
	 *         and the true labels.
	 */
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
		int training_size = DataSet.TESTINGSIZE;
		//int training_size = DataSet.TRAININGSIZE;
		int testing_size = DataSet.TESTINGSIZE;
		DataSet training = new DataSet(
				"/Users/haijieg/workspace/kdd2012/datawithfeature/train3.txt",
				true, training_size);
		DataSet testing = new DataSet(
				"/Users/haijieg/workspace/kdd2012/datawithfeature/test.txt",
				false, testing_size);

		DecimalFormat formatter = new DecimalFormat("###.##");
		LogisticRegression lr = new LogisticRegression();
		double[] steps = {0.001, 0.01, 0.1};
		double lambda = 0.01;
		for (double step : steps) {
			System.err.println("Running step = " + step);
			Weights weights = lr.train(training, lambda, step);
			String ofname = "/Users/haijieg/workspace/kdd2012/experiments/lr/weights_"
					+ step + ".txt";
			BufferedWriter out = new BufferedWriter(new FileWriter(ofname));
			out.write(weights.toString());
			out.close();
			ofname = "/Users/haijieg/workspace/kdd2012/experiments/lr/ctr_"
					+ step + ".txt";
			ArrayList<Double> ctr_prediction = lr.predict(weights, testing);
			out = new BufferedWriter(new FileWriter(ofname));
			String solpath = "/Users/haijieg/workspace/kdd2012/solution/sol.txt";
			double wmse = lr.eval(solpath, ctr_prediction);
			out.write("wmse: " + +wmse + "\n");
			System.out.println("wmse: " + wmse + "\n");
			for (double ctr : ctr_prediction) {
				out.write(formatter.format(ctr) + "\n");
			}
			out.close();
		}
	}
}