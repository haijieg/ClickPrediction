package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import edu.uw.cs.biglearn.clickprediction.analysis.DataSet.DataInstance;
import edu.uw.cs.biglearn.clickprediction.util.FrequencyMap;

public class TokenAnalysis {
	

	/**
	 * Make sure the path is for training data. Otherwise, exception will be thrown.
	 * @param path
	 * @throws Exception
	 */
	public void findPopularInstance(DataSet dataset) throws Exception {
		Scanner sc = new Scanner(new BufferedReader(new FileReader(dataset.path)));
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");		
		DataInstance maxImpressions = null;
		DataInstance maxClicks = null;
		DataInstance maxCTR = null;
		
		while (sc.hasNextLine() && count < dataset.size) {
			String line = sc.nextLine();
			DataInstance instance = dataset.parseLine(line);
			count++;
			
			if (maxImpressions == null || instance.impressions > maxImpressions.impressions)
				maxImpressions = instance;
			
			if (maxClicks == null || instance.clicks > maxImpressions.clicks)
				maxClicks = instance;
			
			if (maxCTR == null || instance.computeCTR() > maxCTR.computeCTR()) {
				maxCTR = instance;
			}
						
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}
		
		System.out.println("Instance with maximum impressions: " + maxImpressions.toString());		
		System.out.println("Instance with maximum clicks: " + maxClicks.toString());		
		System.out.println("Instance with maximum CTR: " + maxCTR.toString());
	}
	

	public Map<Integer, Integer> tokenFrequency(DataSet dataset) throws FileNotFoundException {
		System.err.println("Counting token frequencies");		
		FrequencyMap<Integer> map = new FrequencyMap<Integer>();
		Scanner sc = new Scanner(new BufferedReader(new FileReader(dataset.path)));
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");		
		while (sc.hasNextLine() && count < dataset.size) {
			String line = sc.nextLine();
			DataInstance instance = dataset.parseLine(line);
			count++;			
			int weight = Math.max(instance.impressions, 1);
			for (int token : instance.query)
				map.addOrCreate(token, weight, weight);
			for (int token : instance.title)
				map.addOrCreate(token, weight, weight);
			for (int token : instance.keyword)
				map.addOrCreate(token, weight, weight);
			for (int token : instance.description)
				map.addOrCreate(token, weight, weight);
			if (count % 100000 == 0) {
				System.err.println("Processed " + count + " lines");
			}
		}		
		System.err.println("Done.");				
		return map;
	}
	
	public static void main(String args[]) throws Exception {
		int size = 60000;
		DataSet training = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/train.txt", true, size);
		DataSet testing = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/test.txt",  false, size);
		TokenAnalysis analyzer = new TokenAnalysis();
		
		analyzer.findPopularInstance(training);
		
		Map<Integer, Integer> traintokenfreq = analyzer.tokenFrequency(training);
		Map<Integer, Integer> testtokenfreq = analyzer.tokenFrequency(testing);
		
		System.out.println("Training set unique tokens: " +  traintokenfreq.size());
		System.out.println("Testing set unique tokens: " +  testtokenfreq.size());
		Set<Integer> traintokens = traintokenfreq.keySet();
		Set<Integer> testtokens = testtokenfreq.keySet();
		traintokens.retainAll(testtokens);
		System.out.println("Tokens in common: " +  traintokens.size());
		
		ArrayList<Integer> trainfreqvals =new ArrayList<Integer>(traintokenfreq.values());
		ArrayList<Integer> testfreqvals =new ArrayList<Integer>(testtokenfreq.values());		
		Collections.sort(trainfreqvals);
		Collections.sort(testfreqvals);
		
		BufferedWriter oftrain = new BufferedWriter(new FileWriter("/Users/haijieg/workspace/kdd2012/experiments/tokenfreq_training.txt"));
		oftrain.write(trainfreqvals.toString());
		oftrain.close();
		BufferedWriter oftest = new BufferedWriter(new FileWriter("/Users/haijieg/workspace/kdd2012/experiments/tokenfreq_testing.txt"));
		oftest.write(testfreqvals.toString());
		oftest.close();
	}
}
