package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import edu.uw.cs.biglearn.clickprediction.util.EvalUtil;

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
		Map<Integer, Double> wTokens;
		double wPosition;
		double wDepth;
		double wAge;
		double wGender;
		
		Map<Integer, Integer> accessTime; // keep track of the access timestamp of feature weights.
																			// Using this to do delayed regularization.
		
		public Weights() {
			w0 = wAge = wGender = wDepth = wPosition = 0.0;
			wTokens = new HashMap<Integer, Double>();
			accessTime = new HashMap<Integer, Integer>();
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
			builder.append("Tokens: " + wTokens.toString() + "\n");
			return builder.toString();
		}

		/**
		 * @return the l2 norm of this weight vector.
		 */
		public double l2norm() {
			double l2 = w0 * w0 + wAge * wAge + wGender * wGender
					 				+ wDepth*wDepth + wPosition*wPosition;
			for (double w : wTokens.values())
				l2 += w * w;
			return Math.sqrt(l2);
		}

		/**
		 * @return the l0 norm of this weight vector.
		 */
		public int l0norm() {
			return 4 + wTokens.size();
		}
	}
	


	/**
	 * Helper function to compute inner product w^Tx.
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
		for (int token : instance.tokens) {
			Double w_token = weights.wTokens.get(token);
			if (w_token != null)
				wx += w_token;
		}
		return wx;
	}

	/**
	 * Helper function: update the weights based on the current instance.
	 * @param weights
	 * @param instance
	 * @param step
	 * @param lambda
	 * @param timestamp
	 */
	private void updateWeights(Weights weights, DataInstance instance,
			double step, double lambda, int timestamp) {

		
		
		// compute w0 + <w, x>
		double wx = computeWeightFeatureProduct(weights, instance);
		double exp = Math.exp(wx);
		exp = Double.isInfinite(exp) ? (Double.MAX_VALUE - 1) : exp;
		// compute the gradient
		double grad = (instance.clicked == 1) ? (-1 / (1 + exp)) : (exp / (1 + exp));

		// update weights along the negative gradient
		weights.w0 += -step * grad;
		weights.wAge += -step * (grad * instance.age + lambda * weights.wAge);
		weights.wGender += -step
				* (grad * instance.gender + lambda * weights.wGender);
		weights.wDepth += -step
				* (grad * instance.depth + lambda * weights.wDepth);
		weights.wPosition += -step
				* (grad * instance.position + lambda * weights.wPosition);
		
		for (int token : instance.tokens) {
			Double w = weights.wTokens.get(token);
			if (w == null)
				w = 0.0;
			w += -step * (grad + lambda * w);
			weights.wTokens.put(token, w);
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
			Weights weights,
			int now, double step, double lambda) {
		for (int token : tokens) {
			Integer t = weights.accessTime.get(token);
			if (t != null) {
				double w = weights.wTokens.get(token);
				weights.wTokens.put(token, w * Math.pow((1 - step * lambda), now-t-1));
			}
			weights.accessTime.put(token, now);
		}
	}
	
	
	/**
	 * Train the logistic regression model using the training data and the
	 * hyperparameters. Return the weights, and record the cumulative loss.
	 * 
	 * @param dataset
	 * @param lambda
	 * @param step
	 * @return the weights for the model.
	 */
	public Weights train(DataSet dataset, double lambda, double step, ArrayList<Double> AvgLoss) {
		Weights weights = new Weights();
		int count = 0;
		int loss = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		for (int i = 0; i < 1; i++) {
  		while (dataset.hasNext()) {
  			DataInstance instance = dataset.nextInstance();
  			
    		// Perform delayed regularization
  			if (lambda > 1e-8) {
  	  		performDelayedRegularization(instance.tokens, weights,
  	  				count, step, lambda);
  			}
  			
  			// compute w0 + <w, x>
  			double wx = computeWeightFeatureProduct(weights, instance);
  			double exp = Math.exp(wx);
  			exp = Double.isInfinite(exp) ? (Double.MAX_VALUE - 1) : exp;
  			
  		
  			// compute the gradient
  			double grad = (instance.clicked == 1) ? (-1 / (1 + exp)) : (exp / (1 + exp));
  			
  			// update weights along the negative gradient
  			weights.w0 += -step * grad;
  			weights.wAge += -step * (grad * instance.age + lambda * weights.wAge);
  			weights.wGender += -step
  					* (grad * instance.gender + lambda * weights.wGender);
  			weights.wDepth += -step
  					* (grad * instance.depth + lambda * weights.wDepth);
  			weights.wPosition += -step
  					* (grad * instance.position + lambda * weights.wPosition);  			
  			for (int token : instance.tokens) {
  				Double w = weights.wTokens.get(token);
  				if (w == null)
  					w = 0.0;
  				w += -step * (grad + lambda * w);
  				weights.wTokens.put(token, w);
  			}

  			count++;
  		
  			if (count % 100000 == 0) {
  				System.err.println("Processed " + count + " lines");
  				System.err.println("Average loss: " + (double)loss / count);
  				System.err.println("l2 norm of weights: " + weights.l2norm());
  				// System.err.println("l0 norm of weights: " + weights.l0norm());
  			}
  			
  			// predict the label, record the loss
  		  int click_hat = (exp / (1+exp)) > 0.5 ? 1 : 0;
  		  if (click_hat != instance.clicked)
  		  	loss += 1;
  		  if (count % 100 == 0) {
  		  	AvgLoss.add((double)loss/count);
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

	public static void main(String args[]) throws IOException {
		int training_size = DataSet.TRAININGSIZE;
		int testing_size = DataSet.TESTINGSIZE;
		DataSet training = new DataSet(
				"/Users/haijieg/workspace/kdd2012/simpledata/train.txt",
				true, training_size);
		DataSet testing = new DataSet(
				"/Users/haijieg/workspace/kdd2012/simpledata/test.txt",
				false, testing_size);
		String solpath = "/Users/haijieg/workspace/kdd2012/simpledata/test_label.txt";
		LogisticRegression lr = new LogisticRegression();
		
		double baseline_rmse = EvalUtil.evalBaseLine(solpath, 0.03365528484381977);
		System.out.println("Baseline rmse: " + baseline_rmse);

		DecimalFormat formatter = new DecimalFormat("###.####");
			//double [] steps = {0.001, 0.01, 0.1};
		  //double [] lambdas = {0, 0.01, 0.1};
			double [] steps = {0.05};
			double [] lambdas = {0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005};
		  for (double lambda: lambdas) {
		  	for (double step : steps) {
  			System.err.println("Running step = " + step +", lambda = " + lambda);
  			ArrayList<Double> AvgLoss = new ArrayList<Double>();
  			Weights weights = lr.train(training, lambda, step, AvgLoss);
  			ArrayList<Double> ctr_prediction = lr.predict(weights, testing);
  			double rmse = EvalUtil.eval(solpath, ctr_prediction);
  			System.out.println("rmse: " + rmse + "\n");
  
  			// save the weights and the prediction
  			String outpathbase = "/Users/haijieg/workspace/kdd2012/experiments2/lrreg/";
  			String suffix = "_"+formatter.format(step) + "_"+formatter.format(lambda);
  			BufferedWriter writer = new BufferedWriter(new FileWriter(outpathbase + "weights" + suffix));
  			writer.write("l2 norm: " + weights.l2norm() + "\n");
  			writer.write("l0 norm: " + weights.l0norm() + "\n");  			
  			writer.write(weights.toString());
  			writer.close();
  			
  			writer = new BufferedWriter(new FileWriter(outpathbase + "ctr" + suffix));
  			writer.write("rmse: " + rmse + "\n");
  			for (double ctr : ctr_prediction)
  				writer.write(ctr + "\n");
  			writer.close();
  			
  			writer = new BufferedWriter(new FileWriter(outpathbase + "loss" + suffix));
  			for (double loss : AvgLoss)
  				writer.write(loss + "\n");
  			writer.close();
		  	}
		}
	}
}