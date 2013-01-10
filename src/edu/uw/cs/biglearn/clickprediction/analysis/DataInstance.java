package edu.uw.cs.biglearn.clickprediction.analysis;

import edu.uw.cs.biglearn.clickprediction.util.StringUtil;

/**
 * This class represents an instance of the data.
 * 
 * @author haijieg
 * 
 */
public class DataInstance {
	// Label
	int clicks; // number of clicks, -1 if it is testing data.
	int impressions; // number of impressions, -1 if it is testing data.

	// Feature of the session
	int depth; // depth of the session.
	int[] query; // List of token ids in the query field

	// Feature of the ad
	int position; // position of the ad.
	int[] keyword; // List of token ids in the keyword field
	int[] title; // List of token ids in the title field
	int[] description; // List of token in the description field

	// Feature of the user
	int userid;
	int gender; // user gender indicator -1 for male, 1 for female
	int age;// user age indicator '1' for (0, 12], '2' for (12, 18], '3' for
			// (18, 24], '4' for (24, 30],

	// '5' for (30, 40], and '6' for greater than 40.

	/**
	 * Create a DataInstance from input string.
	 * 
	 * @param line
	 * @param hasLabel
	 *            True if the input string is from training data. False
	 *            otherwise.
	 */
	public DataInstance(String line, boolean hasLabel) {
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
		depth = Integer.valueOf(fields[offset + 0]);
		position = Integer.valueOf(fields[offset + 1]);
		String[] querytokens = fields[offset + 2].split(",");
		query = StringUtil.mapArrayStrToInt(querytokens);
		String[] keywordtokens = fields[offset + 3].split(",");
		keyword = StringUtil.mapArrayStrToInt(keywordtokens);
		String[] titletokens = fields[offset + 4].split(",");
		title = StringUtil.mapArrayStrToInt(titletokens);
		String[] descriptiontokens = fields[offset + 5].split(",");
		description = StringUtil.mapArrayStrToInt(descriptiontokens);
		String[] usertokens = fields[offset + 6].split(",");
		userid = Integer.parseInt(usertokens[0]);
		gender = Integer.parseInt(usertokens[1]);
		gender = (int) ((1 - 1.5) * 2.0); // map gender from {1,2} to {-1, 1}
		age = Integer.parseInt(usertokens[2]);
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		if (clicks >= 0) {
			builder.append(clicks + "|" + impressions + "|");
		}
		builder.append(depth + "|" + position + "|");
		builder.append(StringUtil.implode(query, ",") + "|");
		builder.append(StringUtil.implode(keyword, ",") + "|");
		builder.append(StringUtil.implode(title, ",") + "|");
		builder.append(StringUtil.implode(description, ",") + "|");
		builder.append(userid + "," + gender + "," + age);
		return builder.toString();
	}
}
