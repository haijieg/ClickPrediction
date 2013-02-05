package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.uw.cs.biglearn.clickprediction.util.EvalUtil;

public class LogisticRegressionWithHashing {
	public class Weights {
		double w0;
		double wPosition;
		double wDepth;
		double wAge;
		double wGender;
		double[] wHashedFeature;
		Map<Integer, Integer> accessTime; // keep track of the access timestamp of feature weights.
																			// Using this to do delayed regularization.
		int featuredim;
		
		public Weights(int featuredim) {
			this.featuredim = featuredim;
			w0 = wAge = wGender = wDepth = wPosition = 0.0;
			wHashedFeature = new double[featuredim];
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
			builder.append("HashedFeature: " );
			for (double w: wHashedFeature)
				builder.append(w + " ");
			builder.append("\n");				
			return builder.toString();
		}

		/**
		 * @return the l2 norm of this weight vector.
		 */
		public double l2norm() {
			double l2 = w0 * w0 + wAge * wAge + wGender * wGender
					 				+ wDepth*wDepth + wPosition*wPosition;
			for (double w : wHashedFeature)
				l2 += w * w;
			return Math.sqrt(l2);
		}
	} // end of weight class

	
	/**
	 * Helper function to compute inner product w^Tx.
	 * 
	 * @param weights
	 * @param instance
	 * @return
	 */
	private double computeWeightFeatureProduct(Weights weights,
			HashedDataInstance instance) {
		double wx = weights.w0 + weights.wAge * instance.age + weights.wGender
				* instance.gender + weights.wDepth * instance.depth
				+ weights.wPosition * instance.position;
		for (Map.Entry<Integer, Integer> entry: instance.hashedTextFeature.entrySet()) {
			wx += weights.wHashedFeature[entry.getKey()] * entry.getValue();
		}
		return wx;
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
	private void performDelayedRegularization(Set<Integer> featureids,
			Weights weights,
			int now, double step, double lambda) {
		for (int i : featureids) {
			Integer t = weights.accessTime.get(i);
			if (t != null) {
				double w = weights.wHashedFeature[i];
				weights.wHashedFeature[i]  =  w * Math.pow((1 - step * lambda), now-t-1);
			}
			weights.accessTime.put(i, now);
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
	public Weights train(DataSet dataset, int dim, double lambda, double step, ArrayList<Double> AvgLoss,
			boolean personalized) {
		Weights weights = new Weights(dim);
		int count = 0;
		double loss = 0.0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (dataset.hasNext()) {
			HashedDataInstance instance = dataset.nextHashedInstance(dim,
					personalized);
		
			performDelayedRegularization(instance.hashedTextFeature.keySet(), weights, count, step, lambda);
			
			// compute w0 + <w, x>
			double wx = computeWeightFeatureProduct(weights,
					instance);
			double exp = Math.exp(wx);
			exp = Double.isInfinite(exp) ? (Double.MAX_VALUE - 1) : exp;

			// compute the gradient
			double grad = (instance.clicked == 1) ? (-1 / (1 + exp)) : (exp / (1 + exp));
		
			weights.w0 += -step * grad;
			weights.wAge += -step * (grad * instance.age + lambda * weights.wAge);
			weights.wGender += -step
					* (grad * instance.gender + lambda * weights.wGender);
			weights.wDepth += -step
					* (grad * instance.depth + lambda * weights.wDepth);
			weights.wPosition += -step
					* (grad * instance.position + lambda * weights.wPosition);
			
			// update weights along the negative gradient
			for (Map.Entry<Integer, Integer> entry: instance.hashedTextFeature.entrySet()) {
				int key = entry.getKey();
				weights.wHashedFeature[key] += -step * (grad * entry.getValue() + lambda * weights.wHashedFeature[key]);
			}
			
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
				System.err.println("l2 norm of weights: " + weights.l2norm());
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
			System.err.println("Warning: the real size of the data is less than the input size: "
							+ dataset.size + "<" + count);
		}
		
		// Final sweep for delayed regularization
		Set<Integer> allfeatures = new HashSet<Integer>();
		for (int i = 0; i < weights.wHashedFeature.length; i++)
			allfeatures.add(i);
		performDelayedRegularization(allfeatures, weights, count-1, step, lambda);
		
		System.err.println("Done. Total processed instances: " + count);
		dataset.reset();
		return weights;
	}
	
	/**
	 * Using the weights to predict CTR in for the test dataset.
	 * 
	 * @param weights
	 * @param dataset
	 * @return An array storing the CTR for each datapoint in the test data.
	 */
	public ArrayList<Double> predict(Weights weights, DataSet dataset,
			boolean personalized) {
		ArrayList<Double> ctr = new ArrayList<Double>();
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (dataset.hasNext()) {
			HashedDataInstance instance = dataset.nextHashedInstance(
					weights.featuredim, personalized);
			double wx = computeWeightFeatureProduct(weights,
					instance);
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


	public static void main(String args[]) throws IOException {
		int training_size = DataSet.TRAININGSIZE;
		int testing_size = DataSet.TESTINGSIZE;
		DataSet training = new DataSet(
				"data/train.txt",
				true, training_size);
		DataSet testing = new DataSet(
				"data/test.txt",
				false, testing_size);
		String solpath = "data/test_label.txt";
		LogisticRegressionWithHashing lr = new LogisticRegressionWithHashing();
		
		boolean personal = true; // switch this for personalization

		// filter the testing data that has common users in the training set. 
 		Set<Integer> userInTraining = (new BasicAnalysis()).uniqUsers(training);
  	ArrayList<Boolean> includeList = new ArrayList<Boolean>();			
  	while(testing.hasNext()) {
  		int userid = testing.nextInstance().userid;
  		if (userInTraining.contains(userid))
  			includeList.add(true);
  		else
  			includeList.add(false);
  	}
  	testing.reset();
		
		if (!personal) {
  		double step = 0.01;
  		double lambda = 0.001;
  		int[] dims = {97, 12289, 1572869};
  		//int[] dims = {12289};
  		for (int dim : dims) {
  			System.err.println("Running dim = " + dim);
  			ArrayList<Double> avgLoss  = new ArrayList<Double>();
  			Weights weights = lr.train(training, dim, lambda, step, avgLoss, personal);
  			ArrayList<Double> ctr_prediction = lr.predict(weights, testing,
  				personal);
  			double rmse = EvalUtil.eval(solpath, ctr_prediction);
  			double rmseKnownUser = EvalUtil.evalWithIncludingList(solpath, ctr_prediction, includeList);
  		  System.out.println("rmse: " + rmse + "\n");
  		  System.out.println("rmseKnownUser: " + rmseKnownUser + "\n");
  		  
  		  
  		  // save the weights and the prediction
  			String outpathbase = "experiments/lrhashing/";
  			String suffix = "_"+dim;
  			BufferedWriter writer = new BufferedWriter(new FileWriter(outpathbase + "weights" + suffix));
  			writer.write(weights.toString());
  			writer.close();
  			
  			writer = new BufferedWriter(new FileWriter(outpathbase + "ctr" + suffix));
  			writer.write("rmse: " + rmse + "\n");
  			for (double ctr : ctr_prediction)
  				writer.write(ctr + "\n");
  			writer.close();
  			
  			writer = new BufferedWriter(new FileWriter(outpathbase + "loss" + suffix));
  			for (double loss : avgLoss)
  				writer.write(loss + "\n");
  			writer.close();
    	}
  	} else {  		
  	  		double step = 0.01;
    		double lambda = 0.001;
    		int[] dims = {12289};
    		for (int dim : dims) {
    			System.err.println("Running dim = " + dim);
    			ArrayList<Double> avgLoss  = new ArrayList<Double>();
    			Weights weights = lr.train(training, dim, lambda, step, avgLoss, personal);
    			ArrayList<Double> ctr_prediction = lr.predict(weights, testing,
    				personal);
    			double rmse = EvalUtil.evalWithIncludingList(solpath, ctr_prediction, includeList);
    		  System.out.println("rmse: " + rmse + "\n");
    		  
    		  // save the weights and the prediction
    			String outpathbase = "experiments/lrpersonal/";
    			String suffix = "_"+dim;
    			BufferedWriter writer = new BufferedWriter(new FileWriter(outpathbase + "weights" + suffix));
    			writer.write(weights.toString());
    			writer.close();
    			
    			writer = new BufferedWriter(new FileWriter(outpathbase + "ctr" + suffix));
    			writer.write("rmse: " + rmse + "\n");
    			for (double ctr : ctr_prediction)
    				writer.write(ctr + "\n");
    			writer.close();
    			
    			writer = new BufferedWriter(new FileWriter(outpathbase + "loss" + suffix));
    			for (double loss : avgLoss)
    				writer.write(loss + "\n");
    			writer.close();  		
  	}
  	}
	}
}