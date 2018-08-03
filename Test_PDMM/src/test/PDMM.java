package test;

/**
 * 
 */

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

import javax.jws.soap.SOAPBinding.Use;

import java.util.Map.Entry;

public class PDMM {
	
	public Set<String> wordSet;
	public int numTopic;
	public double alpha, beta, namda;
	public int numIter;
	public int saveStep;
	public ArrayList<PDMM_test> docList;
	public int roundIndex;
	private Random rg;
	public double threshold;
	public double weight;
	public int topWords;
	public int filterSize;
	public String word2idFileName;
	public String similarityFileName;

	public Map<String, Integer> word2id;
	public Map<Integer, String> id2word;
	public Map<Integer, Double> wordIDFMap;
	
	public Map<Integer, Set<Integer>> ZdMap;
	public int[] TdArray;
	
	public Map<Integer, Map<Integer, Double>> docUsefulWords;
	public ArrayList<ArrayList<Integer>> topWordIDList;
	public int vocSize;
	public int numDoc;
	public int maxTd; // the maximum number of topics within a PDMM_test
	private double[][] schema;
	public Map<Integer, int[]> docToWordIDListMap;
	public String initialFileName;  // we use the same initial for DMM-based model
	public double[][] phi;
	private double[] pz;
	private double[][] pdz;
	private double[][] topicProbabilityGivenWord;
	private double[][] pwz;

	public ArrayList<ArrayList<Boolean>> wordGPUFlag; // wordGPUFlag.get(doc).get(wordIndex)
	public Map<Integer, int[]> assignmentListMap; // topic assignment for every PDMM_test
	public ArrayList<ArrayList<Map<Integer, Double>>> wordGPUInfo;

	private double[] nz; // [topic]; nums of words in every topic
	private double[][] nzw; // V_{.k}
	
	private int[] Ck; // Ck[topic]
	private int CkSum;
	public int searchTopK;
	
	private Map<Integer, Map<Integer, Double>> schemaMap;
	
	/*鍒濆鍖栧悇椤瑰弬鏁�*/
	public PDMM(ArrayList<PDMM_test> doc_list, int num_topic, int num_iter, int save_step, double beta,
			double alpha, double namda, double threshold) {
		docList = doc_list;
		numDoc = docList.size();
//		System.out.println("numDoc:"+numDoc);
		numTopic = num_topic;
		this.alpha = alpha;
		numIter = num_iter;
		saveStep = save_step;
		this.threshold = threshold;
		this.beta = beta;
		this.namda = namda;
	}
	
	/*灏嗚瘝璇拰鏂囦欢ID鍒濆鍖�*/
	public boolean loadWordMap(String filename) {
		try {
			FileInputStream fis = new FileInputStream(filename);
			InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
			BufferedReader reader = new BufferedReader(isr);
			String line;
			
			//construct word2id map
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				String[] items = line.split(",");
//				System.out.println(items[0]);
				word2id.put(items[0], Integer.parseInt(items[1]));
				id2word.put(Integer.parseInt(items[1]), items[0]);
			}
//			System.out.println(word2id.size());
			System.out.println("finish read wordmap and the num of word is " + word2id.size());
			return true;
		} catch (Exception e) {
			System.out.println("Error while reading other file:" + e.getMessage());
			e.printStackTrace();
			return false;
		}
	}
	
	/**
	 * Collect the similar words Map, not including the word itself
	 * 
	 * @param filename:
	 *            shcema_similarity filename
	 * @param threshold:
	 *            if the similarity is bigger than threshold, we consider it as
	 *            similar words
	 * @return
	 */

	
	/**
	 * 
	 * @param wordID
	 * @param topic
	 * @return word probability given topic 
	 */

	/**
	 * Get the top words under each topic given current Markov status.
	 * not used in this RatioGPUDMM
	 */


	/**
	 * update the p(z|w) for every iteration
	 */


	public void normalCountWord(Integer topic, int word, Integer flag) {
		nzw[topic][word] += flag;
		nz[topic] += flag;
		//System.out.println("topic:"+topic+"	word:"+word+"	flag:"+flag+"	nzw:"+nzw[topic][word]);
	}

	public void normalCountZd(Set<Integer> Zd, Integer flag){
		for (int topic : Zd){
			Ck[topic] += flag;
			CkSum += flag;
		}
	}
	
	public Set<Integer> getdk_Zd(
			int docID, int[] assignment, int topic){
		Set<Integer> dk = new HashSet<Integer>();
		int[] termIDArray = docToWordIDListMap.get(docID);
		for(int i = 0, length = assignment.length; i < length; i++){
			int z = assignment[i];
			if (z==topic){
				dk.add(termIDArray[i]);
			}
		}
		return dk;
	}
	
	public Map<Integer, Map<Integer, Integer>> getNdkt_Zd(
			int docID, int[] ZdList, int[] assignment){
		Map<Integer, Map<Integer, Integer>> Ndkt = new HashMap<Integer,Map<Integer, Integer>>();
		for(int k : ZdList){
			Ndkt.put(k, new HashMap<Integer, Integer>());
	//		System.out.println(ZdList.length);
		}
		int[] termIDArray = docToWordIDListMap.get(docID);
		for(int i = 0, length = termIDArray.length; i < length; i++){
			int word = termIDArray[i];
			int topic = assignment[i];
		//	System.out.println(topic);
			if (Ndkt.get(topic).containsKey(word)){
				Ndkt.get(topic).put(word, Ndkt.get(topic).get(word)+1);
			}
			else{
				Ndkt.get(topic).put(word, 1);
			}
		}
		return Ndkt;
	}
	
	public Map<Integer, Integer> getNdk_Zd(
			int docID, int[] ZdList, int[] assignment){
		 Map<Integer, Integer> Ndk = new HashMap<Integer,Integer>();
		for(int k : ZdList){
			Ndk.put(k,0);
	//		System.out.println(ZdList.length);
		}
		int[] termIDArray = docToWordIDListMap.get(docID);
		for(int i = 0, length = termIDArray.length; i < length; i++){
			int word = termIDArray[i];
			int topic = assignment[i];
		//	System.out.println(topic);
			Ndk.put(topic, Ndk.get(topic)+1);
		}
		return Ndk;
	}
	
	public void initNewModel() {
		wordGPUFlag = new ArrayList<ArrayList<Boolean>>();
		docToWordIDListMap = new HashMap<Integer, int[]>();
		word2id = new HashMap<String, Integer>();
		id2word = new HashMap<Integer, String>();
		wordIDFMap = new HashMap<Integer, Double>();
		docUsefulWords = new HashMap<Integer, Map<Integer, Double>>();
		wordSet = new HashSet<String>();
		topWordIDList = new ArrayList<ArrayList<Integer>>();
		assignmentListMap = new HashMap<Integer, int[]>();
		wordGPUInfo = new ArrayList<ArrayList<Map<Integer, Double>>>();
		rg = new Random();
		
		ZdMap = new HashMap<Integer, Set<Integer>>(); 
		TdArray = new int[docList.size()];
		// construct vocabulary
		loadWordMap(word2idFileName);

		vocSize = word2id.size();
//		System.out.println("vocsize:"+vocSize);
		phi = new double[numTopic][vocSize];
		pz = new double[numTopic];
		pdz = new double[numDoc][numTopic];
		pwz = new double[vocSize][numTopic];

		schema = new double[vocSize][vocSize];
		topicProbabilityGivenWord = new double[vocSize][numTopic];
		
		PDMM_test doc = null;
		ArrayList<Boolean> docWordGPUFlag = null;
		ArrayList<Map<Integer, Double>> docWordGPUInfo = null;
		int[] termIDArray = null;
		
		for (int i = 0; i < numDoc; i++) {
			doc = docList.get(i);
			assignmentListMap.put(i, new int[doc.words.length]);
			termIDArray = new int[doc.words.length];
			docWordGPUFlag = new ArrayList<Boolean>();
			docWordGPUInfo = 
				new ArrayList<Map<Integer, Double>>();
			for (int j = 0, num_word = doc.words.length; j < num_word; j++) {
				termIDArray[j] = word2id.get(doc.words[j]);
				docWordGPUFlag.add(false); // initial for False for every word
				docWordGPUInfo.add(new HashMap<Integer, Double>());
			}
			wordGPUFlag.add(docWordGPUFlag);
			wordGPUInfo.add(docWordGPUInfo);
			docToWordIDListMap.put(i, termIDArray);
		}

		// init the counter
		nz = new double[numTopic];
		nzw = new double[numTopic][vocSize];
		Ck = new int[numTopic];
		CkSum = 0;
	}


	
	public void init_GSDMM() {
//		schemaMap = loadSchema(similarityFileName, threshold);

		double[] ptd = new double[maxTd];
		double temp_factorial = 1.0;
		for ( int i = 0; i < maxTd; i++ ){
			temp_factorial *= (i+1);
			ptd[i] = Math.pow(namda, (double)(i+1)) * Math.exp(-namda)/temp_factorial;
		}
		
		for (int i = 1; i < maxTd; i++) {
			ptd[i] += ptd[i - 1];
		}
		
		for (int d = 0; d < numDoc; d++) {
			
			double u = rg.nextDouble() * ptd[ptd.length-1];
//			System.out.println("u:"+u);
			int td = -1;
			for (int i = 0, length_ptd = ptd.length; i < length_ptd; i++){
				if(Double.compare(ptd[i], u) >= 0){
					td = i+1;
					break;
				}
			}
			
			assert(td>=1);
			TdArray[d] = td;
//			System.out.println("TD:"+td);
			
			Set<Integer> Zd = new HashSet<Integer>();
			while ( Zd.size() != td ){
				int u_z = rg.nextInt(numTopic);
//				System.out.println("numtopic:"+numTopic);
				Zd.add(u_z);
			}
			ZdMap.put(d, Zd);
			normalCountZd(Zd, +1);
			
			Object[] ZdList = new Object[td];
			ZdList =  Zd.toArray();
			int[] termIDArray = docToWordIDListMap.get(d);
//			System.out.println("length:"+termIDArray.length);
			for (int w = 0, num_word = termIDArray.length; w < num_word; w++){
				int topic_index = rg.nextInt(td);
				int topic = (int) ZdList[topic_index];
				int word = termIDArray[w];
				assignmentListMap.get(d)[w] = topic;
				normalCountWord(topic, word, +1);
			}
		}
		System.out.println("finish init_MU!");
	}

	private static long getCurrTime() {
		return System.currentTimeMillis();
	}
	
	public int[][] get_top10k(){
		int[][] TopK = new int[numDoc][10];
		Map<Integer, Double> Pdz = null;
		for(int d = 0; d < numDoc; d++){
			Pdz = new HashMap<Integer, Double>();
			for(int k = 0; k < numTopic; k++){
				Pdz.put(k, pdz[d][k]);
			}
			
			ArrayList<Entry<Integer, Double>> l = new ArrayList<Entry<Integer, Double>>(Pdz.entrySet());
			Collections.sort(l, new Comparator<Map.Entry<Integer, Double>>() {
				public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2){
//					return (int)((o2.getValue() - o1.getValue())*100000000000000000.0);
					return -Double.compare(o1.getValue(), o2.getValue());
				}
			});
			
			for(int i = 0; i < 10; i++){
				TopK[d][i] = l.get(i).getKey();
			}
		}
		return TopK;
	}
	
	private static int factorial(int n){
		int value = 1;
		while ( n > 0 ){
			value *= n;
			n--;
		}
		
		return value;
	}
	
	
	private int[][] ZdSearchSize(){
		int count = 0;
		int[] boundary = new int[maxTd];
		for ( int i = 0; i < maxTd; i++ ){
			int temp = 1;
			int factorial = factorial(i+1);
			for ( int j = 0; j < i+1; j++ ){
				temp *= (searchTopK - j);
			}
			
			count += temp/factorial;
			boundary[i] = count;
		}
		
		int[][] array = new int[count][];
		for ( int i = 0; i < count; i++ ){
			for ( int j = 0; j < boundary.length; j++ ){
				if ( i < boundary[j] ){
					array[i] = new int[j+1];
					break;
				}
			}
		}
		
		return array;
	}

	public int[][] getTopKTopics(int[][] docTopKTopics){
		Set<Integer> topKTopics = new HashSet<Integer>();
		int minIndex = -1;
		double minValue = 2;
		for(int d = 0; d < numDoc; d++){
			minValue = 2;
			minIndex = -1;
			topKTopics.clear();
			
			for(int k = 0; k < numTopic; k++){
				if ( topKTopics.size() < searchTopK ){
					topKTopics.add(k);
					if ( Double.compare(minValue, pdz[d][k]) > 0 ){
						minValue = pdz[d][k];
						minIndex = k;
					}
				} else {
					if (Double.compare(minValue, pdz[d][k]) < 0) {
						topKTopics.remove(minIndex);
						topKTopics.add(k);
						minIndex = minPDZTopicIndex(d, topKTopics);
						minValue = pdz[d][minIndex];
					}
				}
			}
			
			int index = 0;
			for ( int topic : topKTopics ){
				docTopKTopics[d][index++] = topic;
			}
		}
		
		return docTopKTopics;
	}
	
	
	private int minPDZTopicIndex(int doc, Set<Integer> topics){
		double min = 2;
		int minIndex = -1;
		for ( int topic : topics ){
			if ( Double.compare(min, pdz[doc][topic]) > 0 ){
				min = pdz[doc][topic];
				minIndex = topic;
			}
		}
		
		return minIndex;
	}

	public void run_iteration(String flag) {
		/* Create a new memory block like two dimensional array is very 
		 * expensive in Java. We need to reuse the memory block instead of
		 * creating a new one every time*/
		int[][] topicSettingArray = ZdSearchSize();
		int[][] docTopKTopics = new int[numDoc][searchTopK];
		double[] Ptd_Zd = new double[topicSettingArray.length];
		int[] termIDArray = null;
		int[][] mediateSamples = null;
		
		Map<Integer, int[][]> mediateSampleMap = 
			new HashMap<Integer, int[][]>();
		
		for ( int i = 0; i < numDoc; i++ ){
			termIDArray = docToWordIDListMap.get(i);
			mediateSamples = 
				new int[topicSettingArray.length][termIDArray.length];
			mediateSampleMap.put(i, mediateSamples);
		}
		
		for (int iteration = 1; iteration <= numIter; iteration++) {
//			System.out.println(iteration + "th iteration begin");
			if((iteration%saveStep)==0){
				saveModel(flag+"_iter"+iteration+"_PDMMheu");
			}
			
			long _s = getCurrTime();
			
	//		if don't use heu strategy,please don't Use below three code line
			compute_phi();
			compute_pz();
			compute_pzd();
			
			docTopKTopics = getTopKTopics(docTopKTopics);
			
			for (int s = 0; s < numDoc; s++) {
				termIDArray = docToWordIDListMap.get(s);
				int num_word = termIDArray.length;
				Set<Integer> preZd = ZdMap.get(s);
				normalCountZd(preZd, -1);
				mediateSamples = mediateSampleMap.get(s);
				
				
				topicSettingArray = enumerateTopicSetting(
						topicSettingArray, docTopKTopics[s], maxTd);
				int length_topicSettingArray = topicSettingArray.length;
				
				for(int round = 0; round < length_topicSettingArray; round++){
					int[] topicSetting = topicSettingArray[round];
					int length_topicSetting = topicSetting.length;
					
					for (int w = 0; w < num_word; w++){
						int wordID = termIDArray[w];
						double[] pzDist = new double[length_topicSetting];
						for (int index = 0; index < length_topicSetting; index++) {
							int topic = (int) topicSetting[index];
					//		System.out.println(nzw[topic][wordID]);
							double pz = 1.0 * (nzw[topic][wordID] + beta) / (nz[topic] + vocSize * beta);
							pzDist[index] = pz;
						}

						for (int i = 1; i < length_topicSetting; i++) {
							pzDist[i] += pzDist[i - 1];
						}

						double u = rg.nextDouble() * pzDist[length_topicSetting - 1];
						int newTopic = -1;
						for (int i = 0; i < length_topicSetting; i++) {
							if (Double.compare(pzDist[i], u) >= 0) {
								newTopic = topicSetting[i];
								break;
							}
						}
						// update
						mediateSamples[round][w] = newTopic;
					}
				}   
				
				for (int round = 0; round < length_topicSettingArray; round++){
					int[] topicSetting = topicSettingArray[round];
					int length_topicSetting = topicSetting.length;
					double p1 = Math.pow(namda, topicSetting.length) * Math.exp(-namda);
					double p21 = 1.0;
					double p22 = 1.0;
					
					for(int k : topicSetting){
						p21*= (alpha + Ck[k]);
					}
					
					for(int i = 0; i < length_topicSetting; i++){
						p22 *= (CkSum + numTopic*alpha - i);
					}
					double p2 = p21/p22;
					double p31 = 1.0;
					double p32 = 1.0;
					Map<Integer, Map<Integer, Integer>> Ndkt = 
						getNdkt_Zd(s, topicSetting, mediateSamples[round]);
					Map<Integer, Integer> Ndk = 
						getNdk_Zd(s, topicSetting, mediateSamples[round]);
					for(int k: topicSetting){
						Set<Integer> dk = 
							getdk_Zd(s, mediateSamples[round], k);
					//	System.out.println(dk);
						for(int t: dk){
							for (int i = 0; i < Ndkt.get(k).get(t); i++){
								p31 *= (beta+nzw[k][t]+Ndkt.get(k).get(t)-i);
							}
						}
						for(int j = 0; j < Ndk.get(k); j++){
							p32 *= (nz[k]+vocSize*beta+Ndk.get(k)-j);
						}
						dk.clear();
						
					}
					Ndkt.clear();
					Ndk.clear();
					double p3 = p31/p32;
					Ptd_Zd[round] = p1*p2*p3;
				}
				
				for(int i = 1; i < length_topicSettingArray; i++){
					Ptd_Zd[i]+=Ptd_Zd[i-1];
				}
				
				double u_ptdzd = rg.nextDouble()*Ptd_Zd[length_topicSettingArray-1];
				int new_index = -1;
				for (int i = 0; i < length_topicSettingArray; i++) {
					if (Double.compare(Ptd_Zd[i], u_ptdzd) >= 0) {
						new_index = i;
						break;
					}
				}
				
				TdArray[s] = topicSettingArray[new_index].length;
				preZd.clear();
				for(int k: topicSettingArray[new_index]){
					preZd.add(k);
				}
				
				normalCountZd(preZd, +1);
				System.arraycopy(
						mediateSamples[new_index], 0, 
						assignmentListMap.get(s), 0, mediateSamples[new_index].length);
				
			}
			long _e = getCurrTime();
			System.out.println(iteration + "th iter finished and every iterration costs " + (_e - _s) + "ms! Snippet "
					+ numTopic + " topics round " + roundIndex);
		}
	}
	
	private static int[][] enumerateTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int maxTd) {
		// TODO Auto-generated method stub
		int index = 0;
		if ( maxTd > 0)
			index = enumerateOneTopicSetting(topicSettingArray, topKTopics, index);
		
		if ( maxTd > 1)
		index = enumerateTwoTopicSetting(topicSettingArray, topKTopics, index);
		
		if ( maxTd > 2)
			index = enumerateThreeTopicSetting(topicSettingArray, topKTopics, index);
		
		if ( maxTd > 3)
			index = enumerateFourTopicSetting(topicSettingArray, topKTopics, index);
		
		return topicSettingArray;
	}
	
	private static int enumerateOneTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			topicSettingArray[index++][0] = topKTopics[i];
		}
		
		return index;
	}
	
	private static int enumerateTwoTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			for ( int j = i+1; j < topKTopics.length; j++ ){
				topicSettingArray[index][0] = topKTopics[i];
				topicSettingArray[index++][1] = topKTopics[j];
			}
		}
		
		return index;
	}
	
	private static int enumerateThreeTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			for ( int j = i+1; j < topKTopics.length; j++ ){
				for (int n = j + 1; n < topKTopics.length; n++) {
					topicSettingArray[index][0] = topKTopics[i];
					topicSettingArray[index][1] = topKTopics[j];
					topicSettingArray[index++][2] = topKTopics[n];
				}
			}
		}
		
		return index;
	}
	
	private static int enumerateFourTopicSetting(int[][] topicSettingArray,
			int[] topKTopics, int index){
		for ( int i = 0; i < topKTopics.length; i++ ){
			for ( int j = i+1; j < topKTopics.length; j++ ){
				for (int n = j + 1; n < topKTopics.length; n++) {
					for (int m = n +1; m < topKTopics.length; m++){
					topicSettingArray[index][0] = topKTopics[i];
					topicSettingArray[index][1] = topKTopics[j];
					topicSettingArray[index][2] = topKTopics[n];
					topicSettingArray[index++][3] = topKTopics[m];
					}
				}
			}
		}
		
		return index;
	}

	public void saveModel(String flag) {

		compute_phi();
		compute_pz();
		compute_pzd();
		saveModelPz(flag + "_theta.txt");
		saveModelPhi(flag + "_phi.txt");
		saveModelWords(flag + "_words.txt");
		saveModelZd(flag+"_zd.txt");
		saveModelPdz(flag + "_pdz.txt");
		saveAssign(flag + "_assign.txt");
		saveTermAssign(flag + "_termassign.txt");
	}

	private Boolean saveAssign(String filename) {
		// TODO 閼奉亜濮╅悽鐔稿灇閻ㄥ嫭鏌熷▔鏇炵摠閺嶏拷
		try {
			PrintWriter out = new PrintWriter(filename);
			for (int i = 0; i < numDoc; i++) {
				int[] assignments = assignmentListMap.get(i);
				for (int j = 0; j < assignments.length; j++){
					out.print(assignments[j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving assignments:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}
	
	private Boolean saveTermAssign(String filename) {
		// TODO 閼奉亜濮╅悽鐔稿灇閻ㄥ嫭鏌熷▔鏇炵摠閺嶏拷
		try {
			PrintWriter out = new PrintWriter(filename);
			for (int i = 0; i < numDoc; i++) {
				int[] assignments = assignmentListMap.get(i);
				int[] termIDArray = docToWordIDListMap.get(i);
				for (int j = 0; j < assignments.length; j++){
					out.print(id2word.get(termIDArray[j]) + ":" + assignments[j] + " ");
				}
				out.println();
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving term assignments:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}


	public void compute_phi() {
		for (int i = 0; i < numTopic; i++) {
			double sum = 0.0;
			for (int j = 0; j < vocSize; j++) {
				sum += nzw[i][j];    //鎵�鏈夌殑涓婚涓嬬殑鍗曡瘝鐨勬�诲拰
			}
			
			for (int j = 0; j < vocSize; j++) {
				phi[i][j] = (nzw[i][j] + beta) / (sum + vocSize * beta);  //姣忎釜涓婚涓嬫瘡涓瘝璇殑姒傜巼鍒嗗竷
			}
		}
	}

	public void compute_pz() {
		double sum = 0.0;
		for (int i = 0; i < numTopic; i++) {
			sum += nz[i];   //鎵�鏈変富棰樹笅鐨勫崟璇嶆暟鐩殑鎬诲拰
		}
		
		for (int i = 0; i < numTopic; i++) {
			pz[i] = 1.0 * (nz[i] + alpha) / (sum + numTopic * alpha);  //鍚勪釜涓婚涓瘝鐨勪釜鏁板崰鎬昏瘝鏁扮殑姒傜巼鍒嗗竷
		}
	}

	public void compute_pzd() {
		/** calculate P(z|w) **/
		for (int i = 0; i < vocSize; i++) {
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pz[j] * phi[j][i];   //涓婚-璇嶈鍒嗗竷
				row_sum += pwz[i][j];
			}
			
			for (int j = 0; j < numTopic; j++) {
				pwz[i][j] = pwz[i][j] / row_sum;  //涓婚璇嶈鍒嗗竷锛堟鐜囪〃绀猴級
			}
		}

		for (int i = 0; i < numDoc; i++) {
			int[] doc_word_id = docToWordIDListMap.get(i);
			double row_sum = 0.0;
			for (int j = 0; j < numTopic; j++) {
				pdz[i][j] = 0;
				for (int wordID : doc_word_id) {
					pdz[i][j] += pwz[wordID][j]; //鏂囨。-涓婚鍒嗗竷
				}
				row_sum += pdz[i][j];
			}
			
			for (int j = 0; j < numTopic; j++) {
				pdz[i][j] = pdz[i][j] / row_sum; //鏂囨。涓婚鍒嗗竷姒傜巼琛ㄧ幇褰㈠紡
			}
		}
	}
	
	public Boolean saveModelZd(String filename){
		try {
			PrintWriter out = new PrintWriter(filename);
			int td2 = 0;
			int td3 = 0;
			int td1 = 0;
			int td4 = 0;
			for (int i = 0; i < numDoc; i++) {
				if(ZdMap.get(i).size()==2){
					td2++;
				}
				if(ZdMap.get(i).size()==3){
					td3++;
				}
				if(ZdMap.get(i).size()==1){
					td1++;
				}
				if(ZdMap.get(i).size()==4){
					td4++;
				}
				Iterator it = ZdMap.get(i).iterator();
				while(it.hasNext()){
					out.print(it.next() + " ");
				}
				out.println();
			}
			out.println("td=1:"+td1+" td=2:"+td2+" td=3:"+td3+" td4:"+td4);

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}
	

	public boolean saveModelPdz(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numDoc; i++) {
				for (int j = 0; j < numTopic; j++) {
					out.print(pdz[i][j] + " ");
				}
				out.println();
			}

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving p(z|d) distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPz(String filename) {
		// return false;
		try {
			PrintWriter out = new PrintWriter(filename);

			for (int i = 0; i < numTopic; i++) {
				out.print(pz[i] + " ");
			}
			out.println();

			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saving pz distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelPhi(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename);
			PrintWriter outTW = new PrintWriter("result/Topic_words.txt");
			for (int i = 0; i < numTopic; i++) {
				outTW.println("Topic "+i+" :");
				for (int j = 0; j < vocSize; j++) {
					if(phi[i][j]<1)
					{
						out.print(phi[i][j] + " ");
						if(phi[i][j]>0.01)
						{
//							System.out.println("this is the "+j+" step"+id2word.get(i));
							outTW.println("	"+id2word.get(j)+":"+phi[i][j]);
						}
					}
				}
				outTW.println();
				out.println();
			}
			out.flush();
			out.close();
			outTW.flush();
			outTW.close();
		} catch (Exception e) {
			System.out.println("Error while saving word-topic distribution:" + e.getMessage());
			e.printStackTrace();
			return false;
		}

		return true;
	}

	public boolean saveModelWords(String filename) {
		try {
			PrintWriter out = new PrintWriter(filename, "UTF8");
			for (String word : word2id.keySet()) {
				int id = word2id.get(word);
				out.println(word + "," + id);
			}
			out.flush();
			out.close();
		} catch (Exception e) {
			System.out.println("Error while saveing words list: " + e.getMessage());
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public static void main(String[] args) {
		
		ArrayList<PDMM_test> doc_list = PDMM_test.LoadCorpus("data/tweets.txt");
		//here
//		System.out.print(doc_list.size());
		int numIter = 200, save_step = 200;
		double beta = 0.1;
//		String similarityFileName = "data//qa_word_similarity.txt";
		double weight = 0.1;
		double threshold = 0.7;
		int filterSize = 40;
	
		int[] ls = {0,2,3,4,6,8,9,10};
		
		for (int round = 2; round <= 3;round += 1) {
		for (int l:ls){
	//	for (int maxTd = 4; maxTd <= 4; maxTd +=1){
	//		for (int num_topic = 40; num_topic <= 40; num_topic += 20) {
				int num_topic = 13;
				double alpha = 1.0 * 50 / num_topic;
				double namda = (double)1.0+l/(double)10.0;
				PDMM gsdmm = 
					new PDMM(doc_list, num_topic, numIter, save_step, beta, alpha, namda, threshold);
				gsdmm.word2idFileName = "data/test2id.txt";
				gsdmm.topWords = 100;
				int maxTd = 2;
				
				gsdmm.maxTd = maxTd;
				int Topk = 10;
				gsdmm.searchTopK = Topk; // search size for heuristic search , 
				                          //we don't use heuristic search if we set searchTopK = numTopic
				System.out.println("this is l:"+l);
				gsdmm.filterSize = filterSize;
				gsdmm.roundIndex = round;
//				gsdmm.similarityFileName = similarityFileName;
				gsdmm.weight = weight;
				gsdmm.initNewModel();
				gsdmm.init_GSDMM();
				String flag = round+"round_"+num_topic + "topic_qa_";
				flag = "result/" + flag;
				
				//now GPU is yes , word-filter is yes , heu is yes 
				gsdmm.run_iteration(flag); //remember to check whether GPU and word-filter is used!!
				//System.out.println("the iteration run end！");
				try {
					Thread.sleep(10000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
		System.out.println("the round："+round+" end now!");
		}
		
	}
}


/**
 * Comparator to rank the words according to their probabilities.
 */
class TopicalWordComparator implements Comparator<Integer> {
	private double[] distribution = null;

	public TopicalWordComparator(double[] distribution) {
		this.distribution = distribution;
	}

	@Override
	public int compare(Integer w1, Integer w2) {
		return Double.compare(distribution[w1], distribution[w2]);
	}
}


