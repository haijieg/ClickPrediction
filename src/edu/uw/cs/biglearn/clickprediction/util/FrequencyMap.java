package edu.uw.cs.biglearn.clickprediction.util;

import java.util.HashMap;

public class FrequencyMap<T> extends HashMap<T, Integer> {
	public int addOrCreate(T key, int inc, int def) {
		Integer i = this.get(key);
		if (i == null) {
			this.put(key, def);
			return def;
		} else {
			this.put(key, i + inc);
			return i + inc;
		}
	}

	public static void main(String[] args) {
		FrequencyMap<Integer> map = new FrequencyMap<Integer>();
		map.put(1, 0);
		map.put(2, 0);
		map.addOrCreate(1, 1, 0);
		map.addOrCreate(2, 2, 0);
		map.addOrCreate(3, 3, 0);
		System.err.println(map.toString());
	}
}