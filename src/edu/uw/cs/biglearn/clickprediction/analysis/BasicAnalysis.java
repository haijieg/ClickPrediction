package edu.uw.cs.biglearn.clickprediction.analysis;

import java.util.HashSet;
import java.util.Set;

public class BasicAnalysis {
	/**
	 * Return the uniqe tokens in the dataset.
	 * 
	 * @param dataset
	 * @return
	 */
	public Set<Integer> uniqTokens(DataSet dataset) {
		System.err.println("Find uniqe tokens");
		Set<Integer> tokens = new HashSet<Integer>();
		int count = 0;
		while (dataset.hasNext()) {
			DataInstance instance = dataset.nextInstance();
			for (int token : instance.tokens)
				tokens.add(token);
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}
		dataset.reset();
		return tokens;
	}

	/**
	 * Return the uniqe users in the dataset.
	 * 
	 * @param dataset
	 * @return
	 */
	public Set<Integer> uniqUsers(DataSet dataset) {
		System.err.println("Find uniqe users");
		int count = 0;
		Set<Integer> users = new HashSet<Integer>();
		while (dataset.hasNext()) {
			DataInstance instance = dataset.nextInstance();
			if (instance.userid != 0)
				users.add(instance.userid);

			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}
		dataset.reset();
		return users;
	}

	/**
	 * @return the average CTR for the training set.
	 */
	public double averageCtr(DataSet dataset) {
		System.err.println("Compute average ctr");
		int count = 0;
		int clicks = 0;
		while (dataset.hasNext()) {
			DataInstance instance = dataset.nextInstance();
			clicks += instance.clicked;
			count++;
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}
		dataset.reset();
		return (double) clicks / count;
	}

	public static void main(String args[]) throws Exception {
		DataSet training = new DataSet(
				"/Users/haijieg/workspace/kdd2012/simpledata/train.txt",
				true, DataSet.TRAININGSIZE);
		DataSet testing = new DataSet(
				"/Users/haijieg/workspace/kdd2012/simpledata/test.txt",
				false, DataSet.TESTINGSIZE);
		BasicAnalysis analyzer = new BasicAnalysis();

		System.out.println("Average CTR: " + analyzer.averageCtr(training));

		Set<Integer> traintokens = analyzer.uniqTokens(training);
		Set<Integer> testtokens = analyzer.uniqTokens(testing);

		System.out.println("Training set unique tokens: " + traintokens.size());
		System.out.println("Testing set unique tokens: " + testtokens.size());
		traintokens.retainAll(testtokens);
		System.out.println("Tokens in common: " + traintokens.size());

		Set<Integer> trainUsers = analyzer.uniqUsers(training);
		Set<Integer> testUsers = analyzer.uniqUsers(testing);
		System.out.println("Training set unique users: " + trainUsers.size());
		System.out.println("Testing set unique users: " + testUsers.size());
		trainUsers.retainAll(testUsers);
		System.out.println("Users in common: " + trainUsers.size());
	}
}
