package edu.uw.cs.biglearn.clickprediction.analysis;

import java.util.HashMap;
import java.util.Map;

import edu.uw.cs.biglearn.clickprediction.util.HashUtil;
import edu.uw.cs.biglearn.clickprediction.util.StringUtil;

public class HashedDataInstance {
	// Label
	int clicked; // 0 or 1

	// Feature of the page and ad
	int depth; // depth of the session.
	int position; // position of the ad.

	// Feature of the user
	int userid;
	int gender; // user gender indicator -1 for male, 1 for female
	int age;		// user age indicator '1' for (0, 12], '2' for (12, 18], '3' for
							// (18, 24], '4' for (24, 30],
							// 	'5' for (30, 40], and '6' for greater than 40.
	
	int featuredim;
	Map<Integer, Integer> hashedTextFeature; // map hashed feature key to its value;

	public HashedDataInstance(String line, boolean hasLabel, int dim,
			boolean personal) {
		String[] fields = line.split("\\|");
		int offset = 0;
		if (hasLabel) {
			clicked = Integer.valueOf(fields[0]);
			offset = 1;
		} else {
			clicked = -1;
		}
		depth = Integer.valueOf(fields[offset + 0]);
		position = Integer.valueOf(fields[offset + 1]);
		userid = Integer.valueOf(fields[offset + 2]);
		gender = Integer.valueOf(fields[offset + 3]);
		gender = (int)((gender - 1.5) * 2.0); // map gender from {1,2} to {-1, 1}
		age = Integer.valueOf(fields[offset + 4]);
		
		String[] tokens = fields[offset+5].split(",");

		/**
		 * Fill in your code here to create a hashedFeature.
		 */
		this.featuredim = dim;
		hashedTextFeature = new HashMap<Integer, Integer>();
		for (String token : tokens)
			updateFeature(token, 1);

		if (personal) {
			/**
			 * Extra credit Fill in your code here to for create a hashedFeature
			 * with personalization.
			 */
			updateFeature(userid + "intercept", 1);
			for (String token : tokens)
				updateFeature(userid + token, 1);
		}
	}

	/**
	 * Updates the feature array with a given key and value. 
	 * @param key
	 * @param val
	 */
	private void updateFeature(String key, int val) {
		int hashedkey = HashUtil.hashToRange(key, featuredim);
		int hashedval = HashUtil.hashToSign(key) * val;
		Integer oldvalue = hashedTextFeature.get(hashedkey);
		if (oldvalue == null)
			oldvalue = 0;
		hashedTextFeature.put(hashedkey, oldvalue + hashedval);		
	}
}
