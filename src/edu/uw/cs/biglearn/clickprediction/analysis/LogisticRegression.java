package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import edu.uw.cs.biglearn.clickprediction.analysis.DataSet.DataInstance;

public class LogisticRegression {
	public class Weights {
		double w0;
		Map<Integer, Double> query;
		Map<Integer, Double> title;
		Map<Integer, Double> keyword;		
		Map<Integer, Double> description;		
		public Weights() {
			w0 = 0.0;
			query = new HashMap<Integer, Double>();
			title = new HashMap<Integer, Double>();
			keyword = new HashMap<Integer, Double>();
			description = new HashMap<Integer, Double>();
		}

		@Override
		public String toString() {
			DecimalFormat myFormatter = new DecimalFormat("###.##");
			StringBuilder builder = new StringBuilder();
			builder.append("Intercept: " + myFormatter.format(w0) + "\n");
			builder.append("query: " + query.toString() + "\n");
			builder.append("title: " + title.toString() + "\n");
			builder.append("keyword: " + keyword.toString() + "\n");
			builder.append("description: " + description.toString() + "\n");
			return builder.toString();
		}
		
		public double l2norm() {
			double l2 = w0 * w0;
			for (double w : query.values())
				l2 += w*w;
			for (double w: title.values())
				l2 += w*w;
			for (double w: description.values())
				l2 += w*w;
			for (double w: keyword.values())
				l2 += w*w;
			return Math.sqrt(l2);
		}
	}
	
	private double computeWeightFeatureProduct (Weights weights, DataInstance instance) {
		double wx = weights.w0;
		for (int token: instance.query) {
			Double w_token = weights.query.get(token);
			if (w_token != null)
				wx += w_token;
		}
		for (int token: instance.title) {				
			Double w_token = weights.title.get(token);
			if (w_token != null)
				wx += w_token;
		}
		for (int token: instance.keyword) {
			Double w_token = weights.keyword.get(token);
			if (w_token != null)
				wx += w_token;
		}			
		for (int token: instance.description) {
			Double w_token = weights.description.get(token);
			if (w_token != null)
				wx += w_token;
		}
		return wx;
	}
	
	private void updateWeights(Weights weights,
			DataInstance instance, double step, double grad, double lambda) {
			// update weights along the negative gradient
					weights.w0 += -step * grad;
					for (int token: instance.query) {
						Double w = weights.query.get(token);
						if (w == null) w = 0.0;
						w += -step * (grad + lambda * w * instance.impressions);
						weights.query.put(token, w);
					}
					for (int token: instance.title) {
						Double w = weights.title.get(token);
						if (w == null) w = 0.0;
						w += -step * (grad + lambda * w * instance.impressions);
						weights.title.put(token, w);
					}
					for (int token: instance.keyword) {
						Double w = weights.keyword.get(token);
						if (w == null) w = 0.0;
						w += -step * (grad + lambda * w * instance.impressions);
						weights.keyword.put(token, w);
					}
					for (int token: instance.description) {
						Double w = weights.description.get(token);
						if (w == null) w = 0.0;
						w += -step * (grad + lambda * w * instance.impressions);
						weights.description.put(token, w);
					}
	}
	
	public Weights train (DataSet dataset, double lambda, double step) throws FileNotFoundException {
		Weights weights = new Weights();		
		Scanner sc = new Scanner(new BufferedReader(new FileReader(dataset.path)));
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (sc.hasNextLine() && count < dataset.size) {
			String line = sc.nextLine();
			DataInstance instance = dataset.parseLine(line);			
			// compute w0 + <w, x>
			double wx = computeWeightFeatureProduct(weights, instance);
			double exp = Math.exp(wx);
			exp = Double.isInfinite(exp) ? (Double.MAX_VALUE-1) : exp; 
			int num_positive = instance.clicks;
			int num_negative = instance.impressions - num_positive;
			// compute the gradient
			double grad = num_positive * (-1 / (1+exp)) + (num_negative) * (exp/(1+exp));
			updateWeights(weights, instance, step, grad, lambda);
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
				System.err.println("l2 norm of weights: " + weights.l2norm());
			}
		}
		if (count < dataset.size) {
			System.err.println("Warning: the real size of the data is less than the input size: " + dataset.size + "<" + count);
		}
		System.err.println("Done. Total processed instances: " + count);
		return weights;
	}
	
	public ArrayList<Double> predict (Weights weights, DataSet dataset) throws FileNotFoundException {
		ArrayList<Double> ctr = new ArrayList<Double>();
		
		Scanner sc = new Scanner(new BufferedReader(new FileReader(dataset.path)));
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (sc.hasNextLine() && count < dataset.size) {
			String line = sc.nextLine();			
			DataInstance instance = dataset.parseLine(line);			
			double wx = computeWeightFeatureProduct(weights, instance);
			double exp = Math.exp(wx);
			ctr.add(exp/(1+exp));
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}
		if (count < dataset.size) {
			System.err.println("Warning: the real size of the data is less than the input size: " + dataset.size + "<" + count);
		}
		System.err.println("Done. Total processed instances: " + count);
		return ctr;
	}
	
	public static void main(String args[]) throws IOException {
		DataSet training = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/train.txt", true, DataSet.TRAININGSIZE);
		DataSet testing = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/test.txt", false, DataSet.TESTINGSIZE);
		
		DecimalFormat formatter = new DecimalFormat("###.##");		
		LogisticRegression lr = new LogisticRegression();
		double step = 0.001;
		double[] lambdas = {0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45};
		for (double lambda: lambdas) {
			System.err.println("Running lambda = " + lambda);
			Weights weights = lr.train(training, lambda, step);			
			String ofname = "/Users/haijieg/workspace/kdd2012/experiments/weights_"+formatter.format(lambda)+".txt";
			BufferedWriter out = new BufferedWriter(new FileWriter(ofname));
			out.write(weights.toString());
			out.close();
			ofname = "/Users/haijieg/workspace/kdd2012/experiments/ctr_" + formatter.format(lambda)+".txt";
			ArrayList<Double> ctr_prediction = lr.predict(weights, testing);
			out = new BufferedWriter(new FileWriter(ofname));
			for (double ctr : ctr_prediction) {
				out.write(formatter.format(ctr) + "\n");
			}
			out.close();
		}
	}
}