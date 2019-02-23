package tk.mybatis.springboot.util;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.text.DecimalFormat;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

enum LocalSearchAlgorithm {
    K2, GeneticSearch, HillClimber, LAGDHillClimber, LocalScoreSearchAlgorithm,
    RepeatedHillClimber, SimulatedAnnealing, TabuSearch, TAN
}

class Tuple<T0, T1> {
    private T0 t0;

    private T1 t1;

    Tuple(T0 t0, T1 t1){
        setT0(t0);
        setT1(t1);
    }

    private void setT0(T0 t0) {
        this.t0 = t0;
    }

    public T0 getT0() {
        return t0;
    }

    private void setT1(T1 t1) {
        this.t1 = t1;
    }

    public T1 getT1() {
        return t1;
    }

    @Override
    public boolean equals(Object obj) {
        return ((Tuple)obj).getT0() == this.t0;
    }
}

public class Bayes {
    private static String root_path = new File(".").getAbsoluteFile().getParent()
            + File.separator + "src"+ File.separator + "main"+ File.separator + "java"+ File.separator;
    private Instances originalData;
    private Instances data;
    private List<String> allAttList;
    private JSONObject globalGBN;
    private Map<Integer, Set<Tuple<Integer, Double>>> linksMap;
    private BiMap<String, Integer> nodesMap;
    private void initOriginalData(){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(root_path + "tk\\mybatis\\springboot\\data\\user.arff"));
            this.originalData = new Instances(reader);
        }catch (Exception e) {
            e.printStackTrace();
        }
    }
    private void initData(int discretizeBins){
        try{
            initOriginalData();
            Discretize discretize = new Discretize();
            discretize.setBins(discretizeBins);
            discretize.setInputFormat(this.originalData);
            this.data = Filter.useFilter(this.originalData, discretize);
            this.data.setClassIndex(this.data.numAttributes() - 1);
        }catch (Exception e) {
            e.printStackTrace();
        }
    }
    public Bayes(){
        this(2);
    }

    public Bayes(int discretizeBins){
        initData(discretizeBins);
        this.globalGBN = null;
        allAttList = new ArrayList<>();
        for(int i = 0, numAttributes = this.data.numAttributes(); i < numAttributes; i++){
            allAttList.add(this.data.attribute(i).name());
        }
    }

    /**
     * @param att: attribute name
     * @param type: type of the attribute
     * @return dataList: the JSONArray of data<JSONObject>
     */
    public JSONArray getAttDistribution(String att, String type) {
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
        return dataList;
    }

    /**
     * get global GBN of bayes net
     * @return
     */
    public String getGlobalGBN(){
        return getGlobalGBN("K2");
    }

    /**
     * get global GBN of bayes net with given localSearchAlgorithm
     * @return
     */
    public String getGlobalGBN(String localSearchAlgorithm) {
        if(this.globalGBN != null){
            return this.globalGBN.toJSONString();
        }
        JSONObject gbn = new JSONObject();
        JSONArray nodeList = new JSONArray();
        JSONArray linkList = new JSONArray();
        DecimalFormat df = new DecimalFormat("#0.00");//To use: (String) df.format(Number);
        try {
            BayesNet bn = new BayesNet();

            switch (LocalSearchAlgorithm.valueOf(localSearchAlgorithm)){
                case K2:{
                    K2 algorithm = new K2();
                    algorithm.setMaxNrOfParents(data.numAttributes() - 1);
                    algorithm.setInitAsNaiveBayes(false);
                    bn.setSearchAlgorithm(algorithm);
                }break;
                case GeneticSearch:{
                    GeneticSearch algorithm = new GeneticSearch();
                    bn.setSearchAlgorithm(algorithm);
                }
                case HillClimber:{
                    HillClimber algorithm = new HillClimber();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                case LAGDHillClimber:{
                    LAGDHillClimber algorithm = new LAGDHillClimber();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                case LocalScoreSearchAlgorithm:{
                    LocalScoreSearchAlgorithm algorithm = new LocalScoreSearchAlgorithm();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                case RepeatedHillClimber:{
                    RepeatedHillClimber algorithm = new RepeatedHillClimber();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                case SimulatedAnnealing:{
                    SimulatedAnnealing algorithm = new SimulatedAnnealing();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                case TabuSearch:{
                    TabuSearch algorithm = new TabuSearch();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                case TAN:{
                    TAN algorithm = new TAN();
                    bn.setSearchAlgorithm(algorithm);
                } break;
                default:break;
            }

            SimpleEstimator estimator = new SimpleEstimator();
            estimator.setAlpha(0.5);
            bn.setEstimator(estimator);

            bn.buildClassifier(data);
            bn.buildStructure();
            bn.estimateCPTs();

            int incrId = 0;
            HashMap<String, Integer> eventNoMap = new HashMap<>();
            HashMap<String, Integer> priorMap = new HashMap<>();

            for (int i = 0, nrOfNodes = bn.getNrOfNodes(); i < nrOfNodes; ++i) {
                String att = bn.getNodeName(i);
                for (int j = 0 , cardinality_i = bn.getCardinality(i); j < cardinality_i; ++j) {
                    JSONObject node = new JSONObject();
                    String value = trim_quotation(bn.getNodeValue(i, j));
                    node.put("attName", att);
                    node.put("id", att + ": " + value);
                    node.put("value", 1);
                    node.put("eventNo", incrId);
                    nodeList.add(node);
                    eventNoMap.put(att + ": " + value, incrId++);
                }
            }

            for (int i = 0, numAttributes = data.numAttributes(); i < numAttributes; ++i) {
                String attributeName = data.attribute(i).name();
                for (int j = 0, numInstances = data.numInstances(); j < numInstances; ++j) {
                    String id = attributeName + ": " + trim_quotation(data.instance(j).stringValue(i));
                    if (!priorMap.containsKey(id)) {
                        priorMap.put(id, 0);
                    }
                    priorMap.put(id, priorMap.get(id) + 1);
                }
            }

            int numInstances = data.numInstances();

            for (int iNode = 0, nrOfNodes = bn.getNrOfNodes(); iNode < nrOfNodes; ++iNode) {
                String  att = bn.getNodeName(iNode);
                for (int iValue = 0, cardinalityINode = bn.getCardinality(iNode); iValue < cardinalityINode; ++iValue) {
                    String _val = bn.getNodeValue(iNode, iValue);
                    String val = trim_quotation(_val);
                    String childId = att + ": " + val;

                    for (int iParent = 0, nrOfParents = bn.getNrOfParents(iNode); iParent < nrOfParents; ++iParent) {
                        int parent = bn.getParent(iNode, iParent);
                        String attParent = bn.getNodeName(parent);
                        for (int m = 0, cardinalityParent = bn.getCardinality(parent); m < cardinalityParent; ++m) {
                            String _valParent = bn.getNodeValue(parent, m);
                            String valParent = trim_quotation(_valParent);
                            String parentId = attParent + ": " + valParent;
                            JSONObject link = new JSONObject();
                            double[] cpt = new double[4]; //cpt = [ P(A), P(B), P(A|B), P(A|B') ]
                            cpt[0] = (double)priorMap.get(childId) / numInstances; //P(A)
                            cpt[1] = (double)priorMap.get(parentId) / numInstances; //P(B)

                            int cpt2_p = 0, cpt2_c = 0, cpt3_p = 0, cpt3_c = 0;

                            for (int a = 0; a < numInstances; ++a) {
                                Instance instance = data.instance(a);

                                if (instance.stringValue(data.attribute(attParent)).equals(_valParent)) {
                                    cpt2_p++;
                                    if (instance.stringValue(data.attribute(att)).equals(_val)) {
                                        cpt2_c++;
                                    }
                                } else {
                                    cpt3_p++;
                                    if (instance.stringValue(data.attribute(att)).equals(_val)) {
                                        cpt3_c++;
                                    }
                                }
                            }

                            cpt[2] = (double) cpt2_c / cpt2_p;
                            cpt[3] = (double) cpt3_c / cpt3_p;

                            link.put("source", eventNoMap.get(parentId)); //P(A|B)
                            link.put("target", eventNoMap.get(childId)); //P(A|B')
                            link.put("value", cpt[2]);
                            link.put("cpt", cpt);
                            linkList.add(link);
                        }
                    }
                }
            }

            gbn.put("nodes",nodeList);
            gbn.put("links",linkList);
            this.globalGBN = gbn;
        } catch (Exception e) {
            e.printStackTrace();
        }
        this.linksMap = new HashMap<>();
        this.nodesMap = HashBiMap.create();
        for(Object _node : nodeList){
            JSONObject node = (JSONObject) _node;
            this.nodesMap.put((String)node.get("id"), (Integer)node.get("eventNo"));
        }
        for(Object _link : linkList){
            JSONObject link = (JSONObject) _link;
            Integer source = (Integer)link.get("source");
            Integer target = (Integer)link.get("target");
            Double value = (Double)link.get("value");
            if(this.linksMap.containsKey(source)){
                this.linksMap.get(source).add(new Tuple(target, value));
            }else {
                Set<Tuple<Integer, Double>> tupleSet = new HashSet();
                tupleSet.add(new Tuple(target, value));
                this.linksMap.put(source, tupleSet);
            }
        }
        return gbn.toJSONString();
    }

    public String getRecommendation() {
        return getRecommendation(this.allAttList);
    }

    public String getRecommendation(List<String> attList){
        JSONObject recommendation = new JSONObject();

        long startTime = System.nanoTime();
        recommendation.put("group", getLocalGBN(attList));
        long endTime = System.nanoTime();
        System.out.println("getLocalGBN运行时间： "+(endTime-startTime)/1e9+"s");

        recommendation.put("rec", "");
        return recommendation.toJSONString();
    }

    private JSONArray getLocalGBN(List<String> attList){
        if(this.globalGBN == null){
            getGlobalGBN();
        }
        Map<JSONArray, Integer> localGBNs = new HashMap<>();
        JSONArray jsonLocalGBNs = new JSONArray();
        for (int i = 0, numInstances = data.numInstances(); i < numInstances; i++) {
            Instance instance = data.instance(i);
            JSONArray links = new JSONArray();
            Map<String, Integer> entityEventList = new HashMap();
            for(String att : attList){
                entityEventList.put(att, nodesMap.get(att + ": " + trim_quotation(instance.stringValue(data.attribute(att)))));
            }
            for(String att : attList){
                int eventNo = entityEventList.get(att);
                if(linksMap.containsKey(eventNo)){
                    for(Tuple tuple : linksMap.get(eventNo)){
                        if(entityEventList.containsValue(tuple.getT0())) {
                            JSONObject link = new JSONObject();
                            link.put("source", eventNo);
                            link.put("target", tuple.getT0());
                            link.put("value", tuple.getT1());
                            links.add(link);
                        }
                    }
                }
            }
            if(localGBNs.containsKey(links)){
                Integer currentValue = localGBNs.get(links);
                localGBNs.put(links, currentValue+1);
            }else {
                localGBNs.put(links, 1);
            }
        }

        List<Map.Entry<JSONArray, Integer>> localGBNsList = new ArrayList<>(localGBNs.entrySet());
        Collections.sort(localGBNsList, (o1, o2) -> o2.getValue() - o1.getValue());
//        localGBNsList.sort(Comparator.comparing(Map.Entry<JSONArray, Integer>::getValue).reversed());

        for(Map.Entry<JSONArray, Integer> gbn: localGBNsList){
            JSONObject localGBN = new JSONObject();
            JSONArray links = gbn.getKey();
            Set<Integer> nodesSet = new HashSet<>();
            JSONArray nodes = new JSONArray();
            for(Object _link: links){
                JSONObject link = (JSONObject) _link;
                nodesSet.add((Integer) link.get("source"));
                nodesSet.add((Integer) link.get("target"));
            }
            for(Integer _node : nodesSet){
                JSONObject node = new JSONObject();
                node.put("eventNo",_node);
                node.put("id", nodesMap.inverse().get(_node));
                node.put("value", 1);
                nodes.add(node);
            }
            localGBN.put("num", gbn.getValue());
            localGBN.put("links", links);
            localGBN.put("nodes", nodes);
            jsonLocalGBNs.add(localGBN);
        }
        return jsonLocalGBNs;
    }

    /**
     * remove the quotation on the start and the end if it has
     * @param value
     * @return
     */
    private String trim_quotation(String value){
        if(value.startsWith("\'")){
            value = value.substring(1, value.length()-1);
        }
        return value;
    }

    /**
     * used for function test
     * @param args
     */
    public static void main(String[] args) {
        Bayes bn = new Bayes();
        System.out.println(bn.getRecommendation());
//        System.out.println(bn.getAttDistribution("wei", "numerical"));
//        System.out.println(bn.getAttDistribution("cat", "categorical"));
    }
}

