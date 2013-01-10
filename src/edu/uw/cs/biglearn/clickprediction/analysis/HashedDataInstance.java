package edu.uw.cs.biglearn.clickprediction.analysis;

import edu.uw.cs.biglearn.clickprediction.util.HashUtil;

public class HashedDataInstance {
	// Label
	int clicks;
	int impressions;

	int featuredim; // the size of the hashed feature space.
	int[] hashedFeature; // An array storing the value of the hashed feature.

	public HashedDataInstance(String line, boolean hasLabel, int dim,
			boolean personal) {
		String[] fields = line.split("\\|");
		int offset = 0;
		if (hasLabel) {
			clicks = Integer.valueOf(fields[0]);
			impressions = Integer.valueOf(fields[1]);
			offset = 2;
		} else {
			clicks = -1;
			impressions = -1;
		}
		int depth = Integer.parseInt((fields[offset + 0])); // depth;
		int position = Integer.parseInt((fields[offset + 1])); // position
		String[] querytokens = fields[offset + 2].split(","); // query
		String[] keywordtokens = fields[offset + 3].split(","); // keyword
		String[] titletokens = fields[offset + 4].split(","); // title
		String[] descriptiontokens = fields[offset + 5].split(","); // description

		String[] usertokens = fields[offset + 6].split(",");
		int userid = Integer.parseInt(usertokens[0]); // userid
		int gender = Integer.parseInt(usertokens[1]); // gender
		gender = (int) ((float) gender - 1.5) * 2; // map gender from {1,2} to
													// {-1, 1}
		int age = Integer.parseInt(usertokens[2]); // age

		/**
		 * Fill in your code here to create a hashedFeature.
		 */
		this.featuredim = dim;
		hashedFeature = new int[dim];
		updateFeature("age", age);
		updateFeature("gender", gender);
		updateFeature("depth", depth);
		updateFeature("position", position);
		for (String token : querytokens)
			updateFeature("query" + token, 1);
		for (String token : keywordtokens)
			updateFeature("keyword" + token, 1);
		for (String token : titletokens)
			updateFeature("title" + token, 1);
		for (String token : descriptiontokens)
			updateFeature("description" + token, 1);

		if (personal) {
			/**
			 * Extra credit Fill in your code here to for create a hashedFeature
			 * with personalization.
			 */
			updateFeature(userid + "intercept", 1);
			updateFeature(userid + "age", age);
			updateFeature(userid + "gender", gender);
			updateFeature(userid + "depth", depth);
			updateFeature(userid + "position", position);
			for (String token : querytokens)
				updateFeature(userid + "query" + token, 1);
			for (String token : keywordtokens)
				updateFeature(userid + "keyword" + token, 1);
			for (String token : titletokens)
				updateFeature(userid + "title" + token, 1);
			for (String token : descriptiontokens)
				updateFeature(userid + "description" + token, 1);
		}
	}

	/**
	 * Updates the feature array with a given key and value. 
	 * @param key
	 * @param val
	 */
	private void updateFeature(String key, int val) {
		hashedFeature[HashUtil.hashToRange(key, featuredim)] += HashUtil
				.hashToSign(key) * val;
	}
}
