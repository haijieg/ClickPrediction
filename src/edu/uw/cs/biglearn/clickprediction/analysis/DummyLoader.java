package edu.uw.cs.biglearn.clickprediction.analysis;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

import edu.uw.cs.biglearn.clickprediction.analysis.DataSet.DataInstance;

public class DummyLoader {
	/**
	 * Load data.
	 * @return
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	public void loadAndDoNothing(DataSet dataset) throws IOException, FileNotFoundException{
		Scanner sc = new Scanner(new BufferedReader(new FileReader(dataset.path)));
		int count = 0;
		System.err.println("Loading data from " + dataset.path + " ... ");
		while (sc.hasNextLine() && count < dataset.size) {
			String line = sc.nextLine();
			DataInstance instance = dataset.parseLine(line);
			
			/**
			 * Here goes your code for processing each instance
			 * For example:
			 * 	tracking the max clicks, update token frequency table, or compute gradient for logistic regression...
			 */
			count++;
			if (count % 100000 == 0) {
				System.err.println("Loaded " + count + " lines");
			}
		}
		if (count < dataset.size) {
			System.err.println("Warning: the real size of the data is less than the input size: " + dataset.size + "<" + count);
		}
		System.err.println("Done. Total processed instances: " + count);		
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException {
		int size = 50000;
		DataSet training = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/train.txt", true, size);
		DataSet testing = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/test.txt",  false, size);
		DummyLoader loader = new DummyLoader();
		loader.loadAndDoNothing(training);
		loader.loadAndDoNothing(testing);
	}
}
