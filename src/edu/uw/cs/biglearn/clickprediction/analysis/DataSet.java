package edu.uw.cs.biglearn.clickprediction.analysis;

import edu.uw.cs.biglearn.clickprediction.util.StringUtil;

public class DataSet {
	public class DataInstance {
		int clicks;
		int impressions;
		int depth;
		int position;
		int[] query;
		int[] keyword;
		int[] title;
		int[] description;		
		@Override
		public String toString(){
			StringBuilder builder = new StringBuilder();
			if (clicks >= 0) {
				builder.append(clicks + "|" + impressions +"|");
			}
			builder.append(depth + "|" + position+ "|");
			builder.append(StringUtil.implode(query, ","));
			builder.append(StringUtil.implode(keyword, ","));
			builder.append(StringUtil.implode(title, ","));
			builder.append(StringUtil.implode(description, ","));
			return builder.toString();
		}
		
		public float computeCTR() throws Exception {
			if (clicks < 0) {
				throw new Exception ("Cannot compute CTR on testing data.");
			} else if (impressions == 0) {
				return 0;
			} else {
				return (float)clicks/impressions;
			}
		}
	}
	public static final int TRAININGSIZE = 7485033;
	public static final int TESTINGSIZE= 1016552;
	public String path;
	public boolean hasLabel;
	public int size;
	
	public DataSet (String path, boolean isTraining, int size) {
		this.path = path;
		this.hasLabel = isTraining;
		this.size = size;
	}
	
	public DataInstance parseLine(String line) {
		DataInstance instance = new DataInstance();
		String[] fields = line.split("\\|");
		int offset = 0;
		if (hasLabel) {
			instance.clicks = Integer.valueOf(fields[0]);
			instance.impressions = Integer.valueOf(fields[1]);
			offset = 2;
		} else {
			instance.clicks = -1;
			instance.impressions = -1;
		}
		instance.depth = Integer.valueOf(fields[offset+0]);
		instance.position = Integer.valueOf(fields[offset+1]);
		
		String[] querytokens = fields[offset+2].split(",");			
		instance.query = StringUtil.mapArrayStrToInt(querytokens);
		String[] keywordtokens = fields[offset+3].split(",");
		instance.keyword = StringUtil.mapArrayStrToInt(keywordtokens);
		String[] titletokens = fields[offset+4].split(",");
		instance.title = StringUtil.mapArrayStrToInt(titletokens);
		String[] descriptiontokens = fields[offset+5].split(",");
		instance.description = StringUtil.mapArrayStrToInt(descriptiontokens);
		return instance;
	}
	

	public static void main(String args[]) throws Exception {
		DataSet training = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/train.txt", true, 10000);
		DataSet testing = new DataSet("/Users/haijieg/workspace/kdd2012/datawithfeature/test.txt",  false, 10000);			
	}
}
