package test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;

public class PDMM_test {
	public String [] words;
	public int id;
	public String category;
	
	public PDMM_test(int docid, String category, String [] words){
	    this.id = docid;
	    this.category = category;
	    this.words = words;
	  }
	public static ArrayList<PDMM_test> LoadCorpus(String filename){
	    try{
	      FileInputStream fis = new FileInputStream(filename); 
	      InputStreamReader isr = new InputStreamReader(fis, "UTF-8"); 
	      BufferedReader reader = new BufferedReader(isr);    
	      String line;
	      int id2=0;
	      int k=0;
	      ArrayList<PDMM_test> doc_list = new ArrayList();
	      ArrayList<String> test_repeat=new ArrayList();
	      PrintWriter out = new PrintWriter("data/test2id.txt", "UTF8");
	      while((line = reader.readLine()) != null){
	        line = line.trim(); 
	        String[] items = line.split(" ");
	        //System.out.print(items[0]+" ");
	        int docid = Integer.parseInt(items[0]);
	        String category = items[1];
	        //System.out.print(category+" ");
	        String words_str = items[3].trim();
	        String[] words = words_str.split(",");
	        //System.out.println(words[1]);
	        PDMM_test doc = new PDMM_test(docid, category, words);
	        doc_list.add(doc);
	        for(String word:words)
	        {
	  	      	int test_flag=0;
	        	for(int i=0;i<test_repeat.size();i++)
	        	{
	        		if(word.equals(test_repeat.get(i)))
	        		{
	        			test_flag=1;
//	        			System.out.println("there is a repeat_test "+k+" for word:"+word);
	        			k++;
	        			break;
	        		}
	        	}
	        	if(test_flag==0)
	        	{
	        		out.println(word + "," + id2);
	        		id2++;
	        		test_repeat.add(word);
	        	}
	        }
	      }
	      out.flush();
	      out.close();
	      return doc_list;
	    }
	    catch (Exception e){
	      System.out.println("Error while reading other file:" + e.getMessage());
	      e.printStackTrace();
//	      return false;
	  }
	    return null;
	    
	  }
	public static void main(String[] args) {
	    // TODO 鑷姩鐢熸垚鐨勬柟娉曞瓨鏍�
	    //String [] sarray = {"科技","专科","西安","大学"};
	    //PDMM_test doc = new PDMM_test(1, "院校信息", sarray);
	    ArrayList<PDMM_test> doc_list = PDMM_test.LoadCorpus("data/tweets_after.txt");
	    int s=doc_list.size();
	    for(int i=0;i<s;i++)
	    {
	    	PDMM_test doc=doc_list.get(i);
//	    	System.out.println(doc.words[0]);
	    }
	  }
}
