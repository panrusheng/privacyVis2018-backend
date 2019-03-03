package tk.mybatis.springboot.util;

import com.alibaba.fastjson.JSON;
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

class Truple<T0, T1, T2> {
    private T0 t0;
    private T1 t1;
    private T2 t2;

    Truple(T0 t0, T1 t1, T2 t2){
        setT0(t0);
        setT1(t1);
        setT2(t2);
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
    private void setT2(T2 t2) {
        this.t2 = t2;
    }
    public T2 getT2() {
        return t2;
    }

    @Override
    public boolean equals(Object obj) {
        return ((Truple)obj).getT0() == this.t0;
    }
}

public class Bayes {
    private static String root_path = new File(".").getAbsoluteFile().getParent()
            + File.separator + "src"+ File.separator + "main"+ File.separator + "java"+ File.separator;
    private static final double DELTA = 0.15;
    private static final double MAX_REC_NUM = 3;
    private JSONObject attDiscription;
    private Instances originalData;
    private JSONObject globalGBN;
    private Instances data;
    private List<String> allAttName;
    private Map<String, Boolean> allAttSensitivity;
    private HashMap<String, Integer> priorMap;
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

    private void initAttDiscription(){
        this.attDiscription = new JSONObject();
        this.attDiscription.put("wei" ,  JSON.parseObject("{\"description\" : \"Data adjustment factor\", \"type\": \"numerical\"}"));
        this.attDiscription.put("gen" ,  JSON.parseObject("{\"description\" : \"gender\", \"type\": \"categorical\"}"));
        this.attDiscription.put("cat" ,  JSON.parseObject("{\"description\" : \"Whether he or she is a Catholic believer?\", \"type\": \"categorical\"}"));
        this.attDiscription.put("res" ,  JSON.parseObject("{\"description\" : \"Residence\", \"type\": \"categorical\"}"));
        this.attDiscription.put("sch" ,  JSON.parseObject("{\"description\" : \"School\", \"type\": \"categorical\"}"));
        this.attDiscription.put("fue" ,  JSON.parseObject("{\"description\" : \"Whether the father is unemployed?\", \"type\": \"categorical\"}"));
        this.attDiscription.put("gcs" ,  JSON.parseObject("{\"description\" : \"Whether he or she has five or more GCSEs at grades AC?\", \"type\": \"categorical\"}"));
        this.attDiscription.put("fmp" ,  JSON.parseObject("{\"description\" : \"Whether the father is at least the management?\", \"type\": \"categorical\"}"));
        this.attDiscription.put("lvb" ,  JSON.parseObject("{\"description\" : \"Whether live with parents?\", \"type\": \"categorical\"}"));
        this.attDiscription.put("tra" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep training within six years?\", \"type\": \"numerical\"}"));
        this.attDiscription.put("emp" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep employment within six years?\", \"type\": \"numerical\"}"));
        this.attDiscription.put("jol" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep joblessness within six years?\", \"type\": \"numerical\"}"));
        this.attDiscription.put("fe" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a further education within six years?\", \"type\": \"numerical\"}"));
        this.attDiscription.put("he" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a higher education within six years?\", \"type\": \"numerical\"}"));
        this.attDiscription.put("ascc" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep at school within six years?\", \"type\": \"numerical\"}"));
    }

    public Bayes(){
        initOriginalData();
        initAttDiscription();
        this.globalGBN = null;
    }

    public JSONObject getAttDiscription(){
        return this.attDiscription;
    }

    /**
     * get global GBN of bayes net with given attributes
     * @return
     */
    public String getGBN(List<JSONObject> attList){
        return getGBN("K2", attList);
    }

    /**
     * get gbn and attribute distributions
     * @param localSearchAlgorithm
     * @param attList
     * @return
     */
    public String getGBN(String localSearchAlgorithm, List<JSONObject> attList){
        JSONObject gbn = new JSONObject();
        gbn.put("GBN", getGlobalGBN(localSearchAlgorithm, attList));
        gbn.put("attributes", getAttDistribution(attList));
        return gbn.toJSONString();
    }

    public String getRecommendation(List<String> attList){
        JSONArray recommendationList = new JSONArray();
        JSONArray localGBN = getLocalGBN(attList);
//        Instant start = Instant.now();
        int index = 0;
        for(Object _group : localGBN){
            JSONObject group = (JSONObject) _group;
            JSONObject recommendation = new JSONObject();
            recommendation.put("id", "group_"+(index++));

            JSONObject data = new JSONObject();
            JSONArray nodes = (JSONArray)group.get("nodes");
            for(Object _node : nodes){
                String[] nodeID = ((JSONObject) _node).getString("id").split(": ");
                data.put(nodeID[0], nodeID[1]);
            }
            recommendation.put("data",data);

            JSONArray records = new JSONArray();
            int instanceCounter = 0;
            for(Instance instance : originalData){
                instanceCounter++;
                int nodeCounter = 0;
                JSONArray recordData = new JSONArray();
                for(Object _node : nodes){
                    JSONObject recordDatum  = new JSONObject();
                    nodeCounter++;
                    String[] attEvent = ((JSONObject) _node).getString("id").split(": ");
                    String att = attEvent[0];
                    String event = attEvent[1];
                    String type = (String)((JSONObject) this.attDiscription.get(att)).get("type");
                    if(type.equals("categorical")){
                        if(instance.stringValue(this.originalData.attribute(att)).equals(event)){
                            recordDatum.put("attName", att);
                            recordDatum.put("value", event);
                            recordDatum.put("utility", 1);// fake, to be calculated
                            recordData.add(recordDatum);
                        }else{
                            break;
                        }
                    } else{
                        double numerivalValue = instance.value(this.originalData.attribute(att));
                        String[] min_max = event.split("~");
                        String minString = min_max[0].replaceAll("[\\(\\)\\[\\]]", "");
                        String maxString = min_max[1].replaceAll("[\\(\\)\\[\\]]", "");
                        double minValue = minString.equals("-inf")?Double.MIN_VALUE:Double.valueOf(minString);
                        double maxValue = maxString.equals("inf")?Double.MAX_VALUE:Double.valueOf(maxString);
                        if(numerivalValue >= minValue && numerivalValue <= maxValue){
                            recordDatum.put("attName", att);
                            recordDatum.put("value", numerivalValue);
                            recordDatum.put("utility", 1);// fake, to be calculated
                            recordData.add(recordDatum);
                        }else{
                            break;
                        }
                    }
                }
                if(nodeCounter == nodes.size()){
                    JSONObject record = new JSONObject();
                    record.put("id", "record_"+instanceCounter);
                    record.put("data",recordData);
                    records.add(record);
                }
            }
            recommendation.put("records",records);

            recommendation.put("localGBN", group);

            recommendation.put("recList", getRec(attList, group));

            recommendationList.add(recommendation);
        }
//        Instant end = Instant.now();
//        System.out.println("getRecommendation运行时间： " + Duration.between(start, end).toNanos() + "ns");

        return recommendationList.toJSONString();
    }

    /**
     * get attDistributions & GBN of bayes net with given localSearchAlgorithm & given attributes
     * @return bundle
     */
    private JSONObject getGlobalGBN(String localSearchAlgorithm, List<JSONObject> attList) {
        this.allAttName = new ArrayList();
        this.allAttSensitivity = new HashMap();
        for(JSONObject att : attList){
            this.allAttSensitivity.put((String)att.get("attName"), (Boolean)att.get("sensitive"));
            this.allAttName.add((String)att.get("attName"));
        }
        try{
            Discretize discretize = new Discretize();
            discretize.setBins(2);
            discretize.setInputFormat(originalData);
            this.data = Filter.useFilter(originalData, discretize);
            this.data.setClassIndex(this.data.numAttributes() - 1);

            for(int i = 0, len_i = this.data.numAttributes(); i < len_i; i++){
                if(!this.allAttName.contains(this.data.attribute(i).name())){
                    this.data.deleteAttributeAt(i);
                    i--; len_i--;
                }else{
                    this.data.setClassIndex(i);
                }
            }
        }catch (Exception e) {
            e.printStackTrace();
        }
        JSONObject gbn = new JSONObject();
        JSONArray nodeList = new JSONArray();
        JSONArray linkList = new JSONArray();
        DecimalFormat df = new DecimalFormat("#0.00"); // To use: (String) df.format(Number);
        try {
            BayesNet bn = new BayesNet();
            switch (LocalSearchAlgorithm.valueOf(localSearchAlgorithm)){
                case K2:{
                    K2 algorithm = new K2();
                    algorithm.setMaxNrOfParents(this.data.numAttributes() - 1);
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

            bn.buildClassifier(this.data);
            bn.buildStructure();
            bn.estimateCPTs();

            int incrId = 0;
            HashMap<String, Integer> eventNoMap = new HashMap<>();
            priorMap = new HashMap<>();

            for (int i = 0, nrOfNodes = bn.getNrOfNodes(); i < nrOfNodes; ++i) {
                String att = bn.getNodeName(i);
                for (int j = 0 , cardinality_i = bn.getCardinality(i); j < cardinality_i; ++j) {
                    JSONObject node = new JSONObject();
                    String value = numericFilter(bn.getNodeValue(i, j));
                    node.put("attName", att);
                    node.put("id", att + ": " + value);
                    node.put("value", 1);
                    node.put("eventNo", incrId);
                    nodeList.add(node);
                    eventNoMap.put(att + ": " + value, incrId++);
                }
            }

            for (int i = 0, numAttributes = this.data.numAttributes(); i < numAttributes; ++i) {
                String attributeName = this.data.attribute(i).name();
                for (int j = 0, numInstances = this.data.numInstances(); j < numInstances; ++j) {
                    String id = attributeName + ": " + numericFilter(this.data.instance(j).stringValue(i));
                    if (!priorMap.containsKey(id)) {
                        priorMap.put(id, 0);
                    }
                    priorMap.put(id, priorMap.get(id) + 1);
                }
            }

            int numInstances = this.data.numInstances();

            for (int iNode = 0, nrOfNodes = bn.getNrOfNodes(); iNode < nrOfNodes; ++iNode) {
                String  att = bn.getNodeName(iNode);
                for (int iValue = 0, cardinalityINode = bn.getCardinality(iNode); iValue < cardinalityINode; ++iValue) {
                    String _val = bn.getNodeValue(iNode, iValue);
                    String val = numericFilter(_val);
                    String childId = att + ": " + val;

                    for (int iParent = 0, nrOfParents = bn.getNrOfParents(iNode); iParent < nrOfParents; ++iParent) {
                        int parent = bn.getParent(iNode, iParent);
                        String attParent = bn.getNodeName(parent);
                        for (int m = 0, cardinalityParent = bn.getCardinality(parent); m < cardinalityParent; ++m) {
                            String _valParent = bn.getNodeValue(parent, m);
                            String valParent = numericFilter(_valParent);
                            String parentId = attParent + ": " + valParent;
                            JSONObject link = new JSONObject();
                            double[] cpt = new double[4]; //cpt = [ P(A), P(B), P(A|B), P(A|B') ]
                            cpt[0] = (double)priorMap.get(parentId) / numInstances; //P(A)
                            cpt[1] = (double)priorMap.get(childId) / numInstances; //P(B)

                            int cpt2_p = 0, cpt2_c = 0, cpt3_p = 0, cpt3_c = 0;

                            for (int a = 0; a < numInstances; ++a) {
                                Instance instance = this.data.instance(a);

                                if (instance.stringValue(this.data.attribute(attParent)).equals(_valParent)) {
                                    cpt2_p++;
                                    if (instance.stringValue(this.data.attribute(att)).equals(_val)) {
                                        cpt2_c++;
                                    }
                                } else {
                                    cpt3_p++;
                                    if (instance.stringValue(this.data.attribute(att)).equals(_val)) {
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
        return gbn;
    }

    /**
     *
     * @param attList
     * @param group
     * @return
     */
    private JSONArray getRec(List<String> attList, JSONObject group) {
        JSONArray recList = new JSONArray();
        JSONArray nodes = group.getJSONArray("nodes");
        List<String> normalEvents = new ArrayList<>();
        List<String> sensitiveEvents = new ArrayList<>();
        for(Object _node : nodes){
            String node = ((JSONObject) _node).getString("id");
            if (this.allAttSensitivity.get(node.split(": ")[0])) { //sensitive atts which needs to be calculated
                sensitiveEvents.add(node);
            } else{
                normalEvents.add(node);
            }
        }
        Boolean initProtects = isProtected(normalEvents, sensitiveEvents);
        if(!initProtects) {
            for (int combinationLen = 1, len = normalEvents.size(); combinationLen < len; combinationLen++) {
                if (recList.size() >= MAX_REC_NUM) break;
                for (int i = 0, len_i = normalEvents.size() - combinationLen + 1; i < len_i; i++) {
                    if (recList.size() < MAX_REC_NUM) {
                        List<String> deleteEvents = new ArrayList<>();
                        List<String> subEvents = new ArrayList<>();
                        subEvents.addAll(normalEvents);
                        for (int j = i; j < combinationLen; j++) {
                            deleteEvents.add(normalEvents.get(j));
                        }
                        subEvents.removeAll(deleteEvents);
                        Boolean protects = isProtected(subEvents, sensitiveEvents);
                        if (protects) {
                            JSONObject scheme = new JSONObject();
                            scheme.put("dL", deleteEvents);
                            scheme.put("uL", 1); //fake, to be calculated
                            recList.add(scheme);
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        return recList;
    }

    /**
     * get local GBN of bayes net
     * @return
     */
    private JSONArray getLocalGBN(List<String> attList){
        Map<JSONArray, Integer> localGBNMap = new HashMap<>();
        JSONArray jsonLocalGBN = new JSONArray();
        for (int i = 0, numInstances = data.numInstances(); i < numInstances; i++) {
            Instance instance = data.instance(i);
            JSONArray links = new JSONArray();
            Map<String, Integer> entityEventMap = new HashMap();
            for(String att : attList){
                entityEventMap.put(att, nodesMap.get(att + ": " + numericFilter(instance.stringValue(data.attribute(att)))));
            }
            for(String att : attList){
                int eventNo = entityEventMap.get(att);
                if(linksMap.containsKey(eventNo)){
                    for(Tuple tuple : linksMap.get(eventNo)){
                        if(entityEventMap.containsValue(tuple.getT0())) {
                            JSONObject link = new JSONObject();
                            link.put("source", eventNo);
                            link.put("target", tuple.getT0());
                            link.put("value", tuple.getT1());
                            links.add(link);
                        }
                    }
                }
            }
            if(localGBNMap.containsKey(links)){
                Integer currentValue = localGBNMap.get(links);
                localGBNMap.put(links, currentValue+1);
            }else {
                localGBNMap.put(links, 1);
            }
        }

        List<Map.Entry<JSONArray, Integer>> localGBNList = new ArrayList<>(localGBNMap.entrySet());
        Collections.sort(localGBNList, (o1, o2) -> o2.getValue() - o1.getValue());

        for(Map.Entry<JSONArray, Integer> gbn: localGBNList){
            JSONObject group = new JSONObject();
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
            group.put("num", gbn.getValue());
            group.put("links", links);
            group.put("nodes", nodes);
            jsonLocalGBN.add(group);
        }
        return jsonLocalGBN;
    }

    /**
     * @return dataList: the JSONArray of data<JSONObject>
     */
    private JSONArray getAttDistribution(List<JSONObject> selectAtt) {
        JSONArray attributes = new JSONArray();
        for(JSONObject att: selectAtt){
            String attName = att.getString("attName");
            JSONObject attObj = new JSONObject();
            String type = (String)((JSONObject) this.attDiscription.get(attName)).get("type");
            JSONArray dataList = new JSONArray();
            switch(type){
                case "numerical": {
                    try{
                        Attribute attribute = originalData.attribute(attName);
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
                    Attribute attribute = data.attribute(attName);
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
            attObj.put("attributeName",attName);
            attObj.put("type",type);
            attObj.put("data",dataList);
            attributes.add(attObj);
        }
        return attributes;
    }

    /**
     * calculate the conditional probability : Pr(Es|En1, En2, ...)
     * @param normalEvents: [En1, En2, ...]
     * @param sensitiveEvents: [Es1, Es2, ...]
     * @return isProtected: [protects(Es1), protects(Es2), ...]
     */
    private Boolean isProtected(List<String> normalEvents, List<String> sensitiveEvents) {
        if (this.data == null) return true;

        int numInstances = data.numInstances();
        int numNormalEvents = normalEvents.size();
        int numSensitiveEvents = sensitiveEvents.size();
        double pr_real;
        Boolean isProtected = true;
        double[] numerator = new double[numSensitiveEvents];
        double[] denominator = new double[numSensitiveEvents];
        double[] pr_condition = new double[numSensitiveEvents];
        Arrays.fill(numerator, 0.0);
        Arrays.fill(denominator, 0.0);

        Boolean[] protects = new Boolean[numSensitiveEvents];
        Arrays.fill(protects, true);

        for (Instance instance : data) {
            int i = 0;
            for (; i < numNormalEvents; i++) {
                String[] attEvent = normalEvents.get(i).split(": ");
                String att = attEvent[0];
                String event = attEvent[1];
                if (!numericFilter(instance.stringValue(this.data.attribute(att))).equals(event)) {
                    break;
                }
            }
            if(i == numNormalEvents){
                for(int j = 0; j < numSensitiveEvents; j++){
                    denominator[j]++;
                    String[] attEvent = sensitiveEvents.get(j).split(": ");
                    String att = attEvent[0];
                    String event = attEvent[1];
                    if(numericFilter(instance.stringValue(this.data.attribute(att))).equals(event)){
                        numerator[j]++;
                    }
                }
            }
        }
        //For debug convenience, can be easily accelerated
        try {
            for(int i = 0; i < numSensitiveEvents; i++){
                pr_condition[i] = numerator[i] / denominator[i];
                pr_real = (double)this.priorMap.get(sensitiveEvents.get(i)) / numInstances;
                protects[i] = Math.pow((pr_condition[i] - pr_real), 2) <= Math.pow(DELTA *pr_real, 2);
            }
        } catch (ArithmeticException e) {
            System.out.println("Can not be divided by zero.");
        }
        for(int i = 0; i < numSensitiveEvents; i++){
            isProtected &= protects[i];
        }
        return isProtected;
    }

    /**
     * remove the quotation on the start and the end if it has
     * @param value
     * @return
     */
    private String numericFilter(String value){
        if(value.startsWith("\'")){
            value = value.substring(1, value.length()-1);
            if(value.endsWith(")")){
                return value.replaceAll("-inf[)]", "~inf)");
            }else{
                return value.replaceAll("-inf-", "-inf~");
            }
        }
        else{
            return value;
        }
    }

    /**
     * used for function test
     * @param args
     */
    public static void main(String[] args) {
        Bayes bn = new Bayes();
//        System.out.println(bn.getAttDistribution("wei", "numerical"));
//        System.out.println(bn.getAttDistribution("cat", "categorical"));
    }
}

