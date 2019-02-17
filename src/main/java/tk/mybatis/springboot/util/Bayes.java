package tk.mybatis.springboot.util;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.text.DecimalFormat;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class Bayes {
    private static String root_path = "C:\\Users\\Echizen\\Documents\\work\\privacyVis2018\\src\\main\\java\\";
    private Instances originalData;
    private Instances data;
    private void initOriginalData(){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(root_path + "tk\\mybatis\\springboot\\data\\user.arff"));
            originalData = new Instances(reader);
        }catch (Exception e) {
            e.printStackTrace();
        }
    }
    private void initData(){
        try{
            initOriginalData();
            Discretize discretize = new Discretize();
            discretize.setBins(2);
            discretize.setInputFormat(originalData);
            data = Filter.useFilter(originalData, discretize);
            data.setClassIndex(data.numAttributes() - 1);
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Bayes(){
        initData();
    }
    private String trim_quotation(String value){
        if(value.startsWith("\'")){
            value = value.substring(1, value.length()-1);
        }
        return value;
    }

    public String getGBN() {
        JSONObject gbn = new JSONObject();
        JSONArray nodeList = new JSONArray();
        JSONArray linkList = new JSONArray();
        DecimalFormat df = new DecimalFormat("#0.00");//To use: (String) df.format(Number);
        try {
            BayesNet bn = new BayesNet();
            K2 algorithm = new K2();
            SimpleEstimator estimator = new SimpleEstimator();

            algorithm.setMaxNrOfParents(data.numAttributes() - 1);
            algorithm.setInitAsNaiveBayes(false);
            estimator.setAlpha(0.5);

            bn.setSearchAlgorithm(algorithm);
            bn.setEstimator(estimator);

            bn.buildClassifier(data);
            bn.buildStructure();
            bn.estimateCPTs();

            int incrId = 0;
            HashMap<String, Integer> eventNoMap = new HashMap<>();
            HashMap<String, Integer> priorMap = new HashMap<>();

            for (int i = 0; i < bn.getNrOfNodes(); ++i) {
                String attr = bn.getNodeName(i);
                for (int j = 0; j < bn.getCardinality(i); ++j) {
                    JSONObject node = new JSONObject();
                    String value = trim_quotation(bn.getNodeValue(i, j));
                    node.put("attrName", attr);
                    node.put("id", attr + ": " + value);
                    node.put("value", 1);
                    node.put("eventNo", incrId);
                    nodeList.add(node);
                    eventNoMap.put(attr + ": " + value, incrId++);
                }
            }

            for (int i = 0; i < data.numAttributes(); ++i) {
                for (int j = 0; j < data.numInstances(); ++j) {
                    String id = data.attribute(i).name() + ": " + trim_quotation(data.instance(j).stringValue(i));
                    if (!priorMap.containsKey(id)) {
                        priorMap.put(id, 0);
                    }
                    priorMap.put(id, priorMap.get(id) + 1);
                }
            }

            int numInstances = data.numInstances();

            for (int iNode = 0; iNode < bn.getNrOfNodes(); ++iNode) {
                String  attr = bn.getNodeName(iNode);
                for (int iValue = 0; iValue < bn.getCardinality(iNode); ++iValue) {
                    String _val = bn.getNodeValue(iNode, iValue);
                    String val = trim_quotation(_val);
                    String childId = attr + ": " + val;

                    for (int iParent = 0; iParent < bn.getNrOfParents(iNode); ++iParent) {
                        int parent = bn.getParent(iNode, iParent);
                        String attrParent = bn.getNodeName(parent);
                        for (int m = 0; m < bn.getCardinality(parent); ++m) {
                            String _valParent = bn.getNodeValue(parent, m);
                            String valParent = trim_quotation(_valParent);
                            String parentId = attrParent + ": " + valParent;
                            JSONObject link = new JSONObject();
                            double[] cpt = new double[4];
                            cpt[0] = (double)priorMap.get(childId) / numInstances;
                            cpt[1] = (double)priorMap.get(parentId) / numInstances;

                            int cpt2_p = 0, cpt2_c = 0, cpt3_p = 0, cpt3_c = 0;

                            for (int a = 0; a < data.numInstances(); ++a) {
                                Instance instance = data.instance(a);

                                if (instance.stringValue(data.attribute(attrParent)).equals(_valParent)) {
                                    cpt2_p++;
                                    if (instance.stringValue(data.attribute(attr)).equals(_val)) {
                                        cpt2_c++;
                                    }
                                } else {
                                    cpt3_p++;
                                    if (instance.stringValue(data.attribute(attr)).equals(_val)) {
                                        cpt3_c++;
                                    }
                                }
                            }

                            cpt[2] = (double) cpt2_c / cpt2_p;
                            cpt[3] = (double) cpt3_c / cpt3_p;

                            link.put("source", eventNoMap.get(parentId));
                            link.put("target", eventNoMap.get(childId));
                            link.put("value", cpt[2]);
                            link.put("cpt", cpt);
                            linkList.add(link);
                        }
                    }
                }
            }

            gbn.put("nodes",nodeList);
            gbn.put("links",linkList);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return gbn.toJSONString();
    }

    /**
     * @param att: attribute name
     * @param type: type of the attribute
     * @return dataList: the JSONArray of data<JSONObject>
     */
    public String getAttDistribution(String att, String type) {
        JSONArray dataList = new JSONArray();
        switch(type){
            case "numerical": {
                try{
                    Attribute attribute = originalData.attribute(att);
                    Map<Double, Integer> eventCount = new TreeMap<>();
                    for(int i = 0, numInstances = originalData.numInstances(); i < numInstances; i++){
                        Instance instance = originalData.instance(i);
                        double attKey = instance.value(attribute);
                        if(eventCount.containsKey(attKey)){
                            eventCount.put(attKey, eventCount.get(attKey)+1);
                        } else {
                            eventCount.put(attKey, 1);
                        }
                    }
                    for(Double key : eventCount.keySet()){
                        JSONObject dataItem = new JSONObject();
                        dataItem.put("label", key);
                        dataItem.put("value", eventCount.get(key));
                        dataList.add(dataItem);
                    }
                } catch(Exception e){
                    e.printStackTrace();
                }
            } break;
            case "categorical": {
                Attribute attribute = data.attribute(att);
                Map<String, Integer> eventCount = new HashMap<>();
                for (int i = 0, numInstances = data.numInstances(); i < numInstances; i++) {
                    Instance instance = data.instance(i);
                    String attKey = instance.stringValue(attribute);
                    if(eventCount.containsKey(attKey)){
                        eventCount.put(attKey, eventCount.get(attKey)+1);
                    } else {
                        eventCount.put(attKey, 1);
                    }
                }
                for(String key : eventCount.keySet()){
                    JSONObject dataItem = new JSONObject();
                    dataItem.put("category", key);
                    dataItem.put("value", eventCount.get(key));
                    dataList.add(dataItem);
                }
            } break;
            default: break;
        }
        return dataList.toJSONString();
    }

    public static void main(String[] args) {
        Bayes bn = new Bayes();
        System.out.println(bn.getAttDistribution("wei", "numerical"));
        System.out.println(bn.getAttDistribution("cat", "categorical"));
    }
}

