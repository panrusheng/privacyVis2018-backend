package tk.mybatis.springboot.util.algorithm;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.w3c.dom.Attr;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
    public void setT0(T0 t0) {
        this.t0 = t0;
    }
    public T0 getT0() {
        return t0;
    }
    public void setT1(T1 t1) {
        this.t1 = t1;
    }
    public T1 getT1() {
        return t1;
    }

    @Override
    public boolean equals(Object obj) {
        return ((Tuple)obj).getT0() == this.t0;
    }

    @Override
    public int hashCode() {
        return this.t0.hashCode();
    }

}

class Triple<T0, T1, T2> {
    private T0 t0;
    private T1 t1;
    private T2 t2;

    Triple(T0 t0, T1 t1, T2 t2){
        setT0(t0);
        setT1(t1);
        setT2(t2);
    }
    public void setT0(T0 t0) {
        this.t0 = t0;
    }
    public T0 getT0() {
        return t0;
    }
    public void setT1(T1 t1) {
        this.t1 = t1;
    }
    public T1 getT1() {
        return t1;
    }
    public void setT2(T2 t2) {
        this.t2 = t2;
    }
    public T2 getT2() {
        return t2;
    }

    @Override
    public boolean equals(Object obj) {
        return ((Triple)obj).getT0() == this.t0;
    }

    @Override
    public int hashCode() {
        return this.t0.hashCode();
    }
}

class DataSegment {
    private int start;
    private int end;
    DataSegment(int start, int end){
        this.start = start;
        this.end = end;
    }

    public int getEnd() {
        return end;
    }

    public int getStart() {
        return start;
    }

    @Override
    public boolean equals(Object obj) {
        DataSegment that = (DataSegment)obj;
        return (that.getStart() >= this.start && that.getEnd() <= this.end);
    }
}

public class Bayes {
    private static String root_path = new File(".").getAbsoluteFile().getParent()
            + File.separator + "src"+ File.separator + "main"+ File.separator + "java"+ File.separator;
    private static final int MAX_REC_NUM = 3;
    private static final int GROUP_NUM = 50;
    private static final double LIMIT_RATE = 0.2;
    private DecimalFormat df = new DecimalFormat("#0.00"); // To use: (String) df.format(Number);
    private double riskLimit;
    private JSONObject attDescription;
    private Instances originalData;
    private Instances data;
    private Instances dataAggregated;
    private List<String> allAttName;
    private Map<String, Boolean> allAttSensitivity;
    private Map<String, Double> utilityMap;
    private HashMap<String, Integer> priorMap;
    private Map<Integer, Set<Triple<Integer, Double, JSONArray>>> linksMap; // source -> Set<target>
    private BiMap<String, Integer> nodesMap;
    private Map<String, double[]> attMinMax;
    private Map<String, Tuple<List<Integer>, Boolean>> attGroupList; // Tuple<groupList, isIntNumerical>
    private Map<String, List<Double>> attSplitPoint;
    private List<JSONObject> recommendationList;
    private JSONArray recommendationTrimResult;
    private List<JSONObject> selectionOption = new ArrayList<>();

    public void setRiskLimit(double riskLimit){
        this.riskLimit = riskLimit;
    }

    public Bayes(){
        this("graduate");
    }

    public Bayes(String datasetName){
        initOriginalData(datasetName);
        initAttDescription(datasetName);
    }

    public JSONObject getAttDescription(){
        return this.attDescription;
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
        gbn.put("correlations", getCorrelations());
        return gbn.toJSONString();
    }

    public String editGBN(List<JSONObject> events){
        this.data = new Instances(this.dataAggregated);
        JSONObject gbn = new JSONObject();
        int numAttributes = this.data.numAttributes();
        List<String> editAttName = new ArrayList<>();
        for(JSONObject attInfo : events){
            String attName = attInfo.getString("attrName");
            editAttName.add(attName);
            Attribute attributeOriginal = this.originalData.attribute(attName);
            Attribute attribute = this.data.attribute(attName);
            if(attInfo.containsKey("groups")){ //category
                List<JSONObject> groups = attInfo.getJSONArray("groups").toJavaList(JSONObject.class);
                List<String> newAttributeValues = new ArrayList<>();
                Enumeration e = attribute.enumerateValues();
                while(e.hasMoreElements()){
                    newAttributeValues.add((String)e.nextElement());
                }
                groups.forEach(g->{
                    JSONArray categories = g.getJSONArray("categories");
                    String newEventName = g.getString("name");
                    for(Object eventName : categories){
                        newAttributeValues.remove(eventName);
                    }
                    newAttributeValues.add(newEventName);
                });
                int curAttIndex = numAttributes;
                Attribute newCategoryAttribute = new Attribute("_"+attName, newAttributeValues);
                if (this.data.attribute("_" + attName) != null) {
                    this.data.deleteAttributeAt(this.data.attribute("_" + attName).index());
                    this.data.insertAttributeAt(newCategoryAttribute, numAttributes);
                } else {
                    this.data.insertAttributeAt(newCategoryAttribute, numAttributes++);
                }

                groups.forEach(g->{
                    JSONArray categories = g.getJSONArray("categories");
                    String newEventName = g.getString("name");
                    for (int i = 0, numInstance = this.data.numInstances(); i < numInstance; i++) {
                        Instance instance = this.data.instance(i);
                        if (categories.contains(instance.stringValue(attribute))) {
                            instance.setValue(curAttIndex, newEventName);
                        }
                    }
                });
            } else{ //numeric
                List<Double> newSplitPoint = attInfo.getJSONArray("splitPoints").toJavaList(Double.class);
                double minValue = this.attMinMax.get(attName)[0];
                double maxValue = this.attMinMax.get(attName)[1];
                List<String> attributeValues = new ArrayList<>();
                attributeValues.add("[" + integerTrimEndZero(df.format(minValue)) + "~" + integerTrimEndZero(df.format(minValue + newSplitPoint.get(0))) + "]");
                for(int i = 0, size = newSplitPoint.size()-1; i < size; i++){
                    String category = "(" + integerTrimEndZero(df.format(minValue + newSplitPoint.get(i))) + "~" + integerTrimEndZero(df.format(minValue + newSplitPoint.get(i+1))) + "]";
                    attributeValues.add(category);
                }
                attributeValues.add("(" + integerTrimEndZero(df.format(minValue + newSplitPoint.get(newSplitPoint.size()-1))) + "~" + integerTrimEndZero(df.format(maxValue)) + "]");
                Attribute newCategoryAttribute = new Attribute("_"+attName, attributeValues);
                this.data.insertAttributeAt(newCategoryAttribute, numAttributes++);
                for(int i = 0, numInstance = this.data.numInstances(); i < numInstance; i++) {
                    Instance instance = this.data.instance(i); // to write
                    Instance instanceOrignal = this.originalData.instance(i); // to read
                    double value = instanceOrignal.value(attributeOriginal);
                    int index = 0, size = newSplitPoint.size();
                    for(; index < size; index++){
                        if(value <= newSplitPoint.get(index)){
                            break;
                        }
                    }
                    instance.setValue(numAttributes - 1, index);
                }
            }
            this.data.setClassIndex(numAttributes-1);
            for(int i = 0; i < numAttributes; i++){
                Attribute attribute_i = this.data.attribute(i);
                if(attribute_i.name().equals(attName)){
                    this.data.deleteAttributeAt(i);
                    i--;numAttributes--;
                } else if(attribute_i.name().startsWith("_")){
                    this.data.renameAttribute(i, attribute_i.name().substring(1));
                }
            }
        }
        
        gbn.put("GBN", getGlobalGBN("K2"));
        gbn.put("correlations", getCorrelations());
        return gbn.toJSONString();
    }

    public String getRecommendation(List<JSONObject> links, List<JSONObject> utilityList){
        /* edit link of the gbn */
        for(JSONObject link : links){
            int source = link.getIntValue("source");
            int target = link.getIntValue("target");
            double value = link.getDoubleValue("value");
            JSONArray cpt = link.getJSONArray("cpt");
            Set<Triple<Integer, Double, JSONArray>> targetSet = this.linksMap.get(source);
            Triple oldLink = null;
            for(Triple<Integer, Double, JSONArray> ele: targetSet){
                if(ele.getT0() == target){
                    oldLink = new Triple<>(ele.getT0(), ele.getT1(), ele.getT2());
                }
            }
            Triple newLink = new Triple<>(target, value, cpt);
            targetSet.remove(oldLink);
            targetSet.add(newLink);
            this.linksMap.put(source, targetSet);
        }

        for(JSONObject utility: utilityList){
            this.utilityMap.put(utility.getString("attName"), utility.getDoubleValue("utility"));
        }

        this.recommendationList = new ArrayList<>();
        JSONArray localGBN = getLocalGBN();
        for(Object _group : localGBN){
            JSONObject group = (JSONObject) _group;
            JSONObject recommendation = new JSONObject();

            recommendation.put("localGBN", group);

            JSONObject data = new JSONObject();
            JSONArray nodes = (JSONArray)group.get("nodes");
            for(Object _node : nodes){
                String[] nodeID = JSON.parseObject(_node.toString()).getString("id").split(": ");
                if(!this.allAttSensitivity.get(nodeID[0])){
                    data.put(nodeID[0], nodeID[1]);
                }
            }
            this.allAttName.forEach(attName->{
                if(!this.allAttSensitivity.get(attName) && !data.containsKey(attName)){
                    data.put(attName, "?");
                }
            });
            recommendation.put("data",data);

            Set<String> allAttList = new HashSet<>(this.allAttName);
            nodes.forEach((Object node)->{
                String attName = ((JSONObject) node).getString("id").split(": ")[0];
                allAttList.remove(attName);
            });

            JSONArray records = new JSONArray();
            for(int i = 0, numInstance = this.originalData.numInstances(); i < numInstance; i++){
                Instance instance = this.data.instance(i);
                Instance originalInstance = this.originalData.instance(i);
                JSONArray recordData = new JSONArray();
                int j = 0, numNodes = nodes.size();
                for(; j < numNodes; j++){
                    String[] attEvent = ((JSONObject) nodes.get(j)).getString("id").split(": ");
                    String att = attEvent[0];
                    String event = attEvent[1];
                    String type = this.attDescription.getJSONObject(att).getString("type");
                    Attribute attributeOriginal = this.originalData.attribute(att);
                    if(instance.stringValue(this.data.attribute(att)).equals(event)){
                        if(!this.allAttSensitivity.get(att)) {
                            JSONObject recordDatum  = new JSONObject();
                            recordDatum.put("attName", att);
                            if(type.equals("categorical")) {
                            recordDatum.put("value", event);
                        } else{
                            recordDatum.put("value", originalInstance.value(attributeOriginal));
                        }
                        recordDatum.put("utility", this.utilityMap.get(att));
                        recordData.add(recordDatum);
                        }
                    }else{
                        break;
                    }
                }
                if(j == nodes.size()){
                    allAttList.forEach(att->{
                        String type = this.attDescription.getJSONObject(att).getString("type");
                        JSONObject recordDatum  = new JSONObject();
                        recordDatum.put("attName", att);
                        if(type.equals("categorical")) {
                            recordDatum.put("value", instance.stringValue(this.data.attribute(att)));
                        } else{
                            recordDatum.put("value", originalInstance.value(this.originalData.attribute(att)));
                        }
                        recordDatum.put("utility", this.utilityMap.get(att));
                        recordData.add(recordDatum);
                    });
                    JSONObject record = new JSONObject();
                    record.put("id", i);
                    record.put("data",recordData);
                    records.add(record);
                }
            }
            recommendation.put("records",records);

            Tuple<List<JSONObject>, JSONObject> recResult = getRec(group);
            List<JSONObject> recList = recResult.getT0();
            if(recList.size() > MAX_REC_NUM) {
                recommendation.put("recList", recList.subList(0, MAX_REC_NUM - 1));
            } else{
                recommendation.put("recList", recList);
            }

            JSONObject riskList = recResult.getT1();
            recommendation.put("risk", riskList);

            this.recommendationList.add(recommendation);
        }
        this.recommendationList.sort( (o1, o2) -> {
            boolean risk1 = false, risk2 = false;
            for(Object _delta : o1.getJSONObject("risk").values()){
                double delta = Double.parseDouble(_delta.toString());
                risk1 = risk1 ||  (delta>this.riskLimit);
            }
            for(Object _delta : o2.getJSONObject("risk").values()){
                double delta = Double.parseDouble(_delta.toString());
                risk2 = risk2 ||  (delta>this.riskLimit);
            }
            if(risk1 == risk2){
                return o2.getIntValue("num") - o1.getIntValue("num");
            } else {
                return risk2 ? 1 : -1;
            }
        });
        int index = 0;
        for(JSONObject recommendation : this.recommendationList){
            recommendation.put("id", index++);
        }
        return JSONArray.parseArray(JSON.toJSONString(this.recommendationList)).toJSONString();
    }

    public String getResult(List<JSONObject> options){
        this.selectionOption = options;
        JSONArray results = new JSONArray();
        Map<String, List<Integer>> eventCntMap = new HashMap<>();//contains current numeric data
        Map<String, int[]> numericEventCntMap = new HashMap<>();//original numeric data
        for(String eventName : this.priorMap.keySet()){
            List<Integer> eventCntSeq = new ArrayList<>();
            eventCntSeq.add(this.priorMap.get(eventName));//oriV
            eventCntMap.put(eventName, eventCntSeq);
            String attName = eventName.split(": ")[0];
            String type = this.attDescription.getJSONObject(attName).getString("type");
            if(type.equals("numerical") && !numericEventCntMap.containsKey(attName)){
                int groupNum = this.attGroupList.get(attName).getT0().size();
                int[] groupList = new int[groupNum];
                for(int i = 0; i < groupNum; i++){
                    groupList[i] = this.attGroupList.get(attName).getT0().get(i);
                }
                numericEventCntMap.put(attName, groupList);
            }
        }
        for(int i = 0, len_i = options.size(); i < len_i; i++){
            JSONObject recommendation = this.recommendationList.get(i);
            if(recommendation.getJSONArray("recList").size() == 0) continue;
            JSONObject option = options.get(i);
            boolean flag = option.getBooleanValue("flag");
            JSONArray records = recommendation.getJSONArray("records");
            if(flag){ //整组选择
                int no = option.getIntValue("no");
                JSONArray deleteEvents = recommendation.getJSONArray("recList").getJSONObject(no).getJSONArray("dL");
                int numDeleteEvents = records.size();
                for (Object eventNo : deleteEvents) {
                    String deleteEventName = this.nodesMap.inverse().get(Integer.parseInt(eventNo.toString()));
                    List<Integer> eventCntSeq = eventCntMap.get(deleteEventName);
                    eventCntSeq.add(eventCntSeq.get(eventCntSeq.size() - 1) - numDeleteEvents);//curV
                    String deleteAttName = deleteEventName.split(": ")[0];
                    String type = this.attDescription.getJSONObject(deleteAttName).getString("type");
                    if(type.equals("numerical")){
                        for(Object _record : records){
                            JSONObject record = JSONObject.parseObject(_record.toString());
                            Instance originalInstance = this.originalData.instance(record.getIntValue("id"));
                            double minValue = this.attMinMax.get(deleteAttName)[0];
                            double split = this.attMinMax.get(deleteAttName)[2];
                            double instanceValue = originalInstance.value(this.originalData.attribute(deleteAttName));
                            int groupIndex = 0;
                            int groupNum = this.attGroupList.get(deleteAttName).getT0().size();
                            boolean finalIsInt = this.attGroupList.get(deleteAttName).getT1();
                            for(; groupIndex < groupNum; groupIndex++){
                                double ceilValue;
                                if(finalIsInt){
                                    ceilValue = (int)(minValue+split*groupIndex);
                                } else {
                                    ceilValue = minValue+split*groupIndex;
                                }
                                if(instanceValue <= ceilValue){
                                    break;
                                }
                            }
                            if(groupIndex == groupNum){
                                groupIndex--;
                            }

                            numericEventCntMap.get(deleteAttName)[groupIndex]--;
                        }
                    }
                }
            } else {
                JSONArray selectionList = option.getJSONArray("selectionList");
                for(int j = 0, size = selectionList.size(); j < size; j++){
                    JSONArray selection = selectionList.getJSONArray(j);
                    JSONArray deleteEvents = recommendation.getJSONArray("recList").getJSONObject(j).getJSONArray("dL");
                    int numDeleteEvents = selection.size();
                    for (Object eventNo : deleteEvents) {
                        String deleteEventName = this.nodesMap.inverse().get(Integer.parseInt(eventNo.toString()));
                        List<Integer> eventCntSeq = eventCntMap.get(deleteEventName);
                        eventCntSeq.add(eventCntSeq.get(eventCntSeq.size() - 1) - numDeleteEvents);//curV
                        String deleteAttName = deleteEventName.split(": ")[0];
                        String type = this.attDescription.getJSONObject(deleteAttName).getString("type");
                        if(type.equals("numerical")){
                            for(Object _id : selection){
                                Integer id = Integer.parseInt(_id.toString());
                                Instance originalInstance = this.originalData.instance(id);
                                double minValue = this.attMinMax.get(deleteAttName)[0];
                                double split = this.attMinMax.get(deleteAttName)[2];
                                double instanceValue = originalInstance.value(this.originalData.attribute(deleteAttName));
                                int groupIndex = 0;
                                int groupNum = this.attGroupList.get(deleteAttName).getT0().size();
                                boolean finalIsInt = this.attGroupList.get(deleteAttName).getT1();
                                for(; groupIndex < groupNum; groupIndex++){
                                    double ceilValue;
                                    if(finalIsInt){
                                        ceilValue = (int)(minValue+split*groupIndex);
                                    } else {
                                        ceilValue = minValue+split*groupIndex;
                                    }
                                    if(instanceValue <= ceilValue){
                                        break;
                                    }
                                }
                                if(groupIndex == groupNum){
                                    groupIndex--;
                                }

                                numericEventCntMap.get(deleteAttName)[groupIndex]--;
                            }
                        }
                    }
                }
            }
        }

        for(String attName : this.allAttName){
            if(!this.allAttSensitivity.get(attName)){
                JSONObject result = new JSONObject();
                result.put("attributeName", attName);
                String type = this.attDescription.getJSONObject(attName).getString("type");
                result.put("type", type);
                if(type.equals("categorical")){
                    JSONArray dataList = new JSONArray();
                    List<JSONObject> _dataList = new ArrayList<>();
                    double minRate = Double.POSITIVE_INFINITY;
                    Enumeration e = this.data.attribute(attName).enumerateValues();
                    while(e.hasMoreElements()){
                        String category = (String)e.nextElement();
                        String eventName = attName+": "+category;
                        JSONObject data = new JSONObject();
                        int oriV = eventCntMap.get(eventName).get(0);
                        int curV = eventCntMap.get(eventName).get(eventCntMap.get(eventName).size()-1);
                        data.put("category", category);
                        data.put("oriV", oriV);
                        data.put("curV", curV);
                        double rate = (double)curV / oriV;
                        if(rate < minRate){
                            minRate = rate;
                        }
                        _dataList.add(data);
                    }
                    for(JSONObject _data : _dataList){
                        JSONObject data = new JSONObject();
                        data.put("category", _data.getString("category"));
                        data.put("oriV", _data.getIntValue("oriV"));
                        data.put("curV", _data.getIntValue("curV"));
                        data.put("triV", (int)(_data.getIntValue("oriV") * minRate));
                        dataList.add(data);
                    }
                    result.put("data", dataList);
                } else{ //numerical
                    List<Double> range = new ArrayList<>();
                    range.add(this.attMinMax.get(attName)[0]);
                    range.add(this.attMinMax.get(attName)[1]);
                    result.put("range", range);
                    JSONArray dataList = new JSONArray();
                    double minRate = Double.POSITIVE_INFINITY;
                    List<Integer> oriGroupList = this.attGroupList.get(attName).getT0();
                    int[] curGroupList = numericEventCntMap.get(attName);
//                    Enumeration e = this.data.attribute(attName).enumerateValues();
//                    while(e.hasMoreElements()){
//                        String category = (String)e.nextElement();
//                        String eventName = attName+": "+category;
//                        int oriV = eventCntMap.get(eventName).get(0);
//                        int curV = eventCntMap.get(eventName).get(eventCntMap.get(eventName).size()-1);
//                        double rate = (double)curV / oriV;
//                        if(rate < minRate){
//                            minRate = rate;
//                        }
//                    }
                    for (int i = 0; i < oriGroupList.size(); ++i) {
                        int oriV = oriGroupList.get(i);
                        int curV = curGroupList[i];
                        if (oriV == 0) continue;
                        double rate = (double)curV / oriV;
                        if (rate < minRate) {
                            minRate = rate;
                        }
                    }
                    for(int i = 0; i < oriGroupList.size(); i++){
                        JSONObject data = new JSONObject();
                        int oriV = oriGroupList.get(i);
                        int curV = curGroupList[i];
                        data.put("oriV", oriV);
                        data.put("curV", curV);
                        data.put("triV", (int)(oriV * minRate));
                        dataList.add(data);
                    }
                    result.put("list", dataList);
                }
                results.add(result);
            }
        }
        this.recommendationTrimResult = results;
        return results.toJSONString();
    }

    public JSONArray getTest(String classifier, JSONObject modelOptions, List<String> trimList) {
        Instances proD = this.trimData(trimList, this.selectionOption);

        JSONArray result = new JSONArray();
        for (String attName : allAttName) {
            if (allAttSensitivity.containsKey(attName) && allAttSensitivity.get(attName)) {
                proD.setClass(proD.attribute(attName));
                data.setClass(data.attribute(attName));
                try {
                    JSONArray tRes = Model.test(classifier, data, proD, modelOptions);
                    result.addAll(tRes);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        return result;
    }

    private void initOriginalData(String datasetName){
        try {
            BufferedReader reader = new BufferedReader(new FileReader(root_path + "tk\\mybatis\\springboot\\data\\" + datasetName + ".arff"));
            this.originalData = new Instances(reader);
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initAttDescription(String datasetName){
        this.attDescription = new JSONObject();
        switch (datasetName){
            case "graduate":{
                this.attDescription.put("wei" ,  JSON.parseObject("{\"description\" : \"Data adjustment factor\", \"type\": \"numerical\"}"));
                this.attDescription.put("gen" ,  JSON.parseObject("{\"description\" : \"gender\", \"type\": \"categorical\"}"));
                this.attDescription.put("cat" ,  JSON.parseObject("{\"description\" : \"Whether he or she is a Catholic believer?\", \"type\": \"categorical\"}"));
                this.attDescription.put("res" ,  JSON.parseObject("{\"description\" : \"Residence\", \"type\": \"categorical\"}"));
                this.attDescription.put("sch" ,  JSON.parseObject("{\"description\" : \"School\", \"type\": \"categorical\"}"));
                this.attDescription.put("fue" ,  JSON.parseObject("{\"description\" : \"Whether the father is unemployed?\", \"type\": \"categorical\"}"));
                this.attDescription.put("gcs" ,  JSON.parseObject("{\"description\" : \"Whether he or she has five or more GCSEs at grades AC?\", \"type\": \"categorical\"}"));
                this.attDescription.put("fmp" ,  JSON.parseObject("{\"description\" : \"Whether the father is at least the management?\", \"type\": \"categorical\"}"));
                this.attDescription.put("lvb" ,  JSON.parseObject("{\"description\" : \"Whether live with parents?\", \"type\": \"categorical\"}"));
                this.attDescription.put("tra" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep training within six years?\", \"type\": \"numerical\"}"));
                this.attDescription.put("emp" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep employment within six years?\", \"type\": \"numerical\"}"));
                this.attDescription.put("jol" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep joblessness within six years?\", \"type\": \"numerical\"}"));
                this.attDescription.put("fe" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a further education within six years?\", \"type\": \"numerical\"}"));
                this.attDescription.put("he" ,  JSON.parseObject("{\"description\" : \"How many month he or she pursue a higher education within six years?\", \"type\": \"numerical\"}"));
                this.attDescription.put("ascc" ,  JSON.parseObject("{\"description\" : \"How many month he or she keep at school within six years?\", \"type\": \"numerical\"}"));
            } break;
            case "home": {
                this.attDescription.put("claim3years" ,  JSON.parseObject("{\"description\" : \"Whether there was loss in last 3 years\", \"type\": \"categorical\"}"));
                this.attDescription.put("empStatus" ,  JSON.parseObject("{\"description\" : \"Client's professional status\", \"type\": \"categorical\"}"));
                this.attDescription.put("busUse" ,  JSON.parseObject("{\"description\" : \"Commercial use indicator\", \"type\": \"categorical\"}"));
                this.attDescription.put("adBuild" ,  JSON.parseObject("{\"description\" : \"Building coverage - Self damage\", \"type\": \"categorical\"}"));
                this.attDescription.put("riskRate" ,  JSON.parseObject("{\"description\" : \"Geographical Classification of Risk - Building\", \"type\": \"numerical\"}"));
                this.attDescription.put("insurSum" ,  JSON.parseObject("{\"description\" : \"Assured Sum - Building\", \"type\": \"numerical\"}"));
                this.attDescription.put("grantYear" ,  JSON.parseObject("{\"description\" : \"Bonus Malus - Building\", \"type\": \"numerical\"}"));
                this.attDescription.put("specInsur" ,  JSON.parseObject("{\"description\" : \"Assured Sum - Valuable Personal Property\", \"type\": \"numerical\"}"));
                this.attDescription.put("specPrem" ,  JSON.parseObject("{\"description\" : \"Premium - Personal valuable items\", \"type\": \"numerical\"}"));
                this.attDescription.put("birthYear" ,  JSON.parseObject("{\"description\" : \"Year of birth of the client\", \"type\": \"numerical\"}"));
                this.attDescription.put("marStatus" ,  JSON.parseObject("{\"description\" : \"Marital status of the client\", \"type\": \"categorical\"}"));
                this.attDescription.put("sex" ,  JSON.parseObject("{\"description\" : \"Customer sex\", \"type\": \"categorical\"}"));
                this.attDescription.put("alarm" ,  JSON.parseObject("{\"description\" : \"Appropriate alarm\", \"type\": \"categorical\"}"));
                this.attDescription.put("lock" ,  JSON.parseObject("{\"description\" : \"Appropriate lock\", \"type\": \"categorical\"}"));
                this.attDescription.put("bedroom" ,  JSON.parseObject("{\"description\" : \"Number of bedrooms\", \"type\": \"numerical\"}"));
                this.attDescription.put("roofCon" ,  JSON.parseObject("{\"description\" : \"Code of the type of construction of the roof\", \"type\": \"categorical\"}"));
                this.attDescription.put("wallCon" ,  JSON.parseObject("{\"description\" : \"Code of the type of construction of the wall\", \"type\": \"categorical\"}"));
                this.attDescription.put("flood" ,  JSON.parseObject("{\"description\" : \"House susceptible to floods\", \"type\": \"categorical\"}"));
                this.attDescription.put("unocc" ,  JSON.parseObject("{\"description\" : \"Number of days unoccupied\", \"type\": \"numerical\"}"));
                this.attDescription.put("neigh" ,  JSON.parseObject("{\"description\" : \"Vigils of proximity present\", \"type\": \"categorical\"}"));
                this.attDescription.put("occStatus" ,  JSON.parseObject("{\"description\" : \"Occupancy status\", \"type\": \"categorical\"}"));
                this.attDescription.put("subside" ,  JSON.parseObject("{\"description\" : \"Subsidence indicator (relative downwards motion of the surface )\", \"type\": \"categorical\"}"));
                this.attDescription.put("safeInstall" ,  JSON.parseObject("{\"description\" : \"Safe installs\", \"type\": \"categorical\"}"));
                this.attDescription.put("yearBuild" ,  JSON.parseObject("{\"description\" : \"Year of construction\", \"type\": \"numerical\"}"));
                this.attDescription.put("payment" ,  JSON.parseObject("{\"description\" : \"Method of payment\", \"type\": \"categorical\"}"));
                this.attDescription.put("polStatus" ,  JSON.parseObject("{\"description\" : \"Police status\", \"type\": \"categorical\"}"));
                this.attDescription.put("lastPrem" ,  JSON.parseObject("{\"description\" : \"Premium - Total for the previous year\", \"type\": \"numerical\"}"));
                this.attDescription.put("garBefore" ,  JSON.parseObject("{\"description\" : \"Option Gardens included before 1st renewal\", \"type\": \"categorical\"}"));
                this.attDescription.put("garPost" ,  JSON.parseObject("{\"description\" : \"Option Gardens included after 1st renewal\", \"type\": \"categorical\"}"));
                this.attDescription.put("keyBefore" ,  JSON.parseObject("{\"description\" : \"Option Replacement of keys included before 1st renewal\", \"type\": \"categorical\"}"));
                this.attDescription.put("keyAfter" ,  JSON.parseObject("{\"description\" : \"Option Replacement of keys included after 1st renewal\", \"type\": \"categorical\"}"));
            } break;
            case "student": {
                this.attDescription.put("gender" ,  JSON.parseObject("{\"description\" : \"Student's gender\", \"type\": \"categorical\"}"));
                this.attDescription.put("nation" ,  JSON.parseObject("{\"description\" : \"Student's nationality\", \"type\": \"categorical\"}"));
                this.attDescription.put("placeofBirth" ,  JSON.parseObject("{\"description\" : \"Student's Place of birth\", \"type\": \"categorical\"}"));
                this.attDescription.put("stage" ,  JSON.parseObject("{\"description\" : \"Educational level student belongs\", \"type\": \"categorical\"}"));
                this.attDescription.put("grade" ,  JSON.parseObject("{\"description\" : \"Grade student belongs\", \"type\": \"numerical\"}"));
                this.attDescription.put("section" ,  JSON.parseObject("{\"description\" : \"Classroom student belongs\", \"type\": \"categorical\"}"));
                this.attDescription.put("topic" ,  JSON.parseObject("{\"description\" : \"Course topic\", \"type\": \"categorical\"}"));
                this.attDescription.put("semester" ,  JSON.parseObject("{\"description\" : \"School year semester\", \"type\": \"categorical\"}"));
                this.attDescription.put("relation" ,  JSON.parseObject("{\"description\" : \"Parent responsible for student\", \"type\": \"categorical\"}"));
                this.attDescription.put("raiseHand" ,  JSON.parseObject("{\"description\" : \"How many times the student raises his/her hand on classroom\", \"type\": \"numerical\"}"));
                this.attDescription.put("visitRes" ,  JSON.parseObject("{\"description\" : \"How many times the student visits a course content\", \"type\": \"numerical\"}"));
                this.attDescription.put("announce" ,  JSON.parseObject("{\"description\" : \"How many times the student checks the new announcements\", \"type\": \"numerical\"}"));
                this.attDescription.put("discuss" ,  JSON.parseObject("{\"description\" : \"How many times the student participate on discussion groups\", \"type\": \"numerical\"}"));
                this.attDescription.put("satisfy" ,  JSON.parseObject("{\"description\" : \"The degree of parent satisfaction from school\", \"type\": \"categorical\"}"));
                this.attDescription.put("absence" ,  JSON.parseObject("{\"description\" : \"The number of absence days for each student\", \"type\": \"categorical\"}"));
                this.attDescription.put("anwser" ,  JSON.parseObject("{\"description\" : \"If parents answer the survey\", \"type\": \"categorical\"}"));
                this.attDescription.put("class" ,  JSON.parseObject("{\"description\" : \"The classification of students decided by numerical interva\", \"type\": \"categorical\"}"));
            }break;
            default: break;
        }
    }

    /**
     * get attDistributions & GBN of bayes net with given localSearchAlgorithm & given attributes
     * @return bundle
     */
    private JSONObject getGlobalGBN(String localSearchAlgorithm, List<JSONObject> attList) {
        this.allAttName = new ArrayList<>();
        this.allAttSensitivity = new HashMap<>();
//        for(JSONObject att : attList){
        attList.parallelStream().forEach(att->{
            String attName = (String)att.get("attName");
            this.allAttName.add(attName);
            this.allAttSensitivity.put(attName, (Boolean)att.get("sensitive"));
        });
        initUtilityMap();
        dataAggregate();
        return getGlobalGBN(localSearchAlgorithm);
    }

    private JSONObject getGlobalGBN(String localSearchAlgorithm){
        JSONObject gbn = new JSONObject();
        JSONArray nodeList = new JSONArray();
        JSONArray linkList = new JSONArray();
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
                final int fi = i;
                for (int j = 0, numInstances = this.data.numInstances(); j < numInstances; ++j) {
                    Instance instance = this.data.instance(j);
                    String id = attributeName + ": " + numericFilter(instance.stringValue(fi));
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
                            cpt[0] = (priorMap.containsKey(parentId) ? (double)priorMap.get(parentId) : 0.0) / numInstances; //P(A)
                            cpt[1] = (priorMap.containsKey(childId) ? (double)priorMap.get(childId) : 0.0) / numInstances; //P(B)

                            int[] cpt2_p = {0}, cpt2_c = {0}, cpt3_p = {0}, cpt3_c = {0};

//                            for (int a = 0; a < numInstances; ++a) {
//                                Instance instance = this.data.instance(a);
                            this.data.forEach(instance -> {

                                if (instance.stringValue(this.data.attribute(attParent)).equals(_valParent)) {
                                    cpt2_p[0]++;
                                    if (instance.stringValue(this.data.attribute(att)).equals(_val)) {
                                        cpt2_c[0]++;
                                    }
                                } else {
                                    cpt3_p[0]++;
                                    if (instance.stringValue(this.data.attribute(att)).equals(_val)) {
                                        cpt3_c[0]++;
                                    }
                                }
                            });

                            cpt[2] = (double) cpt2_c[0] / cpt2_p[0];
                            cpt[3] = (double) cpt3_c[0] / cpt3_p[0];

                            link.put("source", eventNoMap.get(parentId)); //P(A|B)
                            link.put("target", eventNoMap.get(childId)); //P(A|B')
                            link.put("value", cpt[2]-cpt[1]);
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
        this.linksMap = new HashMap<>();
        this.nodesMap = HashBiMap.create();
        for(Object _node : nodeList){
            JSONObject node = (JSONObject) _node;
            this.nodesMap.put((String)node.get("id"), (Integer)node.get("eventNo"));
        }
        for(Object _link : linkList){
            JSONObject link = (JSONObject) _link;
            int source = link.getIntValue("source");
            int target = link.getIntValue("target");
            double value = link.getDoubleValue("value");
            JSONArray cpt = link.getJSONArray("cpt");
            if(this.linksMap.containsKey(source)){
                this.linksMap.get(source).add(new Triple<>(target, value, cpt));
            }else {
                Set<Triple<Integer, Double, JSONArray>> tripleSet = new HashSet<>();
                tripleSet.add(new Triple<>(target, value, cpt));
                this.linksMap.put(source, tripleSet);
            }
        }
        return gbn;
    }

    /**
     *
     * @param group
     * @return
     */
    private Tuple<List<JSONObject>, JSONObject> getRec(JSONObject group) {
        int numInstances = data.numInstances();
        List<JSONObject> recList = new LinkedList<>();
        JSONObject riskList = new JSONObject();
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
        double[] risk = getRisk(normalEvents, sensitiveEvents);
        for(int i = 0, n = risk.length; i < n; i++){
            riskList.put(sensitiveEvents.get(i), risk[i]);
        }
        List<DataSegment> deletedEventsSegment = new ArrayList<>();
        Boolean initProtects = isProtected(risk);
        if(!initProtects) {
            for (int combinationLen = 1, len = normalEvents.size(); combinationLen <= len; combinationLen++) {
                for (int i = 0, len_i = normalEvents.size() - combinationLen; i <= len_i; i++) {
                    DataSegment toDeleteDataSegment = new DataSegment(i, i+combinationLen-1);
                    if(deletedEventsSegment.contains(toDeleteDataSegment)) break;
                    List<String> deleteEvents = new ArrayList<>();
                    List<String> subEvents = new ArrayList<>();
                    subEvents.addAll(normalEvents);
                    for (int j = 0; j < combinationLen; j++) {
                        deleteEvents.add(normalEvents.get(i+j));
                    }
                    subEvents.removeAll(deleteEvents);
                    Boolean protects = isProtected(subEvents, sensitiveEvents);
                    if (protects) {
                        deletedEventsSegment.add(toDeleteDataSegment);
                        JSONObject scheme = new JSONObject();
                        double utilityLoss = 0.0;
                        for(String event : deleteEvents){
                            utilityLoss += this.utilityMap.get(event.split(": ")[0]) * (numInstances - (this.priorMap.get(event)))/numInstances;
                        }
                        scheme.put("dL", deleteEvents.stream().map(this.nodesMap::get).collect(Collectors.toList()));
                        scheme.put("uL", utilityLoss);
                        recList.add(scheme);
                    }
                }
            }
        }
        recList.sort( (o1, o2) -> {
            double v1 = o1.getDoubleValue("uL");
            double v2 = o2.getDoubleValue("uL");
            if(v2 > v1){
                return -1;
            } else if(v2 < v1){
                return 1;
            } else{
                return 0;
            }
        });
        return new Tuple<>(recList, riskList);
    }

    /**
     * get local GBN of bayes net
     * @return
     */
    private JSONArray getLocalGBN() {
        Map<JSONArray, Integer> localGBNMap = new HashMap<>();
        List<JSONObject> localGBN = new ArrayList<>();
        for(Instance instance : this.data){
            JSONArray links = new JSONArray();
            Map<String, Integer> entityEventMap = new HashMap<>();
            for(String att : this.allAttName){
                entityEventMap.put(att, this.nodesMap.get(att + ": " + numericFilter(instance.stringValue(this.data.attribute(att)))));
            }
            for(String att : this.allAttName){
                int eventNo = entityEventMap.get(att);
                if(this.linksMap.containsKey(eventNo)){
                    for(Triple triple : this.linksMap.get(eventNo)){
                        if(entityEventMap.containsValue(triple.getT0())) {
                            JSONObject link = new JSONObject();
                            link.put("source", eventNo);
                            link.put("target", triple.getT0());
                            link.put("value", triple.getT1());
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

        localGBNMap.forEach((JSONArray links, Integer num)->{
            JSONObject group = new JSONObject();
            Set<Integer> nodesSet = new HashSet<>();
            JSONArray nodes = new JSONArray();
            links.forEach((Object _link)->{
                JSONObject link = JSON.parseObject(_link.toString());
                nodesSet.add((Integer) link.get("source"));
                nodesSet.add((Integer) link.get("target"));
            });
            nodesSet.forEach(_node->{
                JSONObject node = new JSONObject();
                node.put("eventNo",_node);
                node.put("id", this.nodesMap.inverse().get(_node));
                node.put("value", 1);
                nodes.add(node);
            });
            group.put("num", num);
            group.put("links", links);
            group.put("nodes", nodes);
            localGBN.add(group);
        });
        localGBN.sort( (o1, o2) -> o2.getIntValue("num") - o1.getIntValue("num"));
        JSONArray jsonLocalGBN = JSONArray.parseArray(JSON.toJSONString(localGBN));
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
            String type = this.attDescription.getJSONObject(attName).getString("type");
            JSONArray dataList = new JSONArray();
            switch(type){
                case "numerical": {
                    try{
                        JSONArray range = new JSONArray();
                        range.add(this.attMinMax.get(attName)[0]);
                        range.add(this.attMinMax.get(attName)[1]);
                        attObj.put("range", range);
                        attObj.put("list", this.attGroupList.get(attName).getT0());
                        attObj.put("splitPoints", this.attSplitPoint.get(attName));
                    } catch(Exception e){
                        e.printStackTrace();
                    }
                } break;
                case "categorical": {
                    Attribute attribute = this.data.attribute(attName);
                    Map<String, Integer> eventCount = new HashMap<>();
                    for (int i = 0, numInstances = this.data.numInstances(); i < numInstances; i++) {
                        Instance instance = this.data.instance(i);
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
                    attObj.put("data",dataList);
                } break;
                default: break;
            }
            attObj.put("attributeName",attName);
            attObj.put("type",type);
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
    private double[] getRisk(List<String> normalEvents, List<String> sensitiveEvents) {
        int numInstances = this.data.numInstances();
        int numNormalEvents = normalEvents.size();
        int numSensitiveEvents = sensitiveEvents.size();
        double[] risk = new double[numSensitiveEvents];
        Arrays.fill(risk, 0.0);
        if(numNormalEvents == 0){
            return risk;
        }

        double pr_real;
        double[] numerator = new double[numSensitiveEvents];
        double[] denominator = new double[numSensitiveEvents];
        double[] pr_condition = new double[numSensitiveEvents];
        Arrays.fill(numerator, 0.0);
        Arrays.fill(denominator, 0.0);

//        for (Instance instance : this.data) {
        this.data.forEach(instance -> {
            for(int i = 0; i < numSensitiveEvents; i++){
                String sensitiveEvent = sensitiveEvents.get(i);
                Integer sensitiveEventNo = this.nodesMap.get(sensitiveEvent);
                int j = 0;
                for (; j < numNormalEvents; j++) {
                    String normalEvent = normalEvents.get(j);
                    Integer normalEventNo = this.nodesMap.get(normalEvent);
                    if((this.linksMap.containsKey(normalEventNo)
                            && this.linksMap.get(normalEventNo).contains(new Triple<>(sensitiveEventNo,0.0,null))
                            )||
                            (this.linksMap.containsKey(sensitiveEventNo)
                            && this.linksMap.get(sensitiveEventNo).contains(new Triple<>(normalEventNo,0.0,null))
                            )) {
                        String[] attEvent = normalEvent.split(": ");
                        String att = attEvent[0];
                        String event = attEvent[1];
                        if (!numericFilter(instance.stringValue(this.data.attribute(att))).equals(event)) {
                            break;
                        }
                    }
                }
                if(j == numNormalEvents){
                    denominator[i]++;
                    String[] attEvent = sensitiveEvent.split(": ");
                    String att = attEvent[0];
                    String event = attEvent[1];
                    if(numericFilter(instance.stringValue(this.data.attribute(att))).equals(event)){
                        numerator[i]++;
                    }
                }
            }
        });
        try {
            for(int i = 0; i < numSensitiveEvents; i++){
                pr_condition[i] = numerator[i] / denominator[i];
                pr_real = (double)this.priorMap.get(sensitiveEvents.get(i)) / numInstances;
                risk[i] = Math.abs(pr_condition[i] - pr_real);
            }
        } catch (ArithmeticException e) {
            System.out.println("Can not be divided by zero.");
        }
        return risk;
    }

    private Boolean isProtected(List<String> normalEvents, List<String> sensitiveEvents) {
        double[] risk = getRisk(normalEvents, sensitiveEvents);
        return isProtected(risk);
    }

    private Boolean isProtected(double[] risk){
        boolean isProtected = true;
        for (int i = 0, numSensitiveEvents = risk.length; i < numSensitiveEvents; i++) {
            isProtected &= (risk[i] <= this.riskLimit);
        }
        return isProtected;
    }

    /**
     * remove the quotation on the start and the end if it has
     * @param value
     * @return
     */
    private String numericFilter(String value) {
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

    private String integerTrimEndZero(String datum){
        if(datum.endsWith(".00")){
            return datum.substring(0, datum.length()-3);
        }
        return datum;
    }

    private void initUtilityMap(){
        this.utilityMap = new HashMap<>();
        for(String attName : this.allAttName){
            this.utilityMap.put(attName, -1.0);
        }
    }

    private void dataAggregate(){
        this.attMinMax = new HashMap<>();
        this.attGroupList = new HashMap<>();
        this.attSplitPoint = new HashMap<>();

        for(String attName : this.allAttName){
            String type = this.attDescription.getJSONObject(attName).getString("type");
            if(type.equals("numerical")){
                double[] minMax = new double[3];
                minMax[0] = Double.POSITIVE_INFINITY; //min
                minMax[1] = Double.NEGATIVE_INFINITY; //max
                minMax[2] = 0.0; //split: (max - min) / groupNum
                this.attMinMax.put(attName, minMax);

                List<Double> splitPoint = new ArrayList<>();
                this.attSplitPoint.put(attName, splitPoint);
            }
        }

        this.attMinMax.forEach((String attName, double[] value)->{
//            for (int i = 0, numInstance = this.originalData.numInstances(); i < numInstance; i++) {
//                Instance instance = this.originalData.instance(i);
            Attribute attributeoriginal = this.originalData.attribute(attName);
            this.originalData.forEach(instance -> {
                double attEventValue = instance.value(attributeoriginal);
                if (attEventValue < value[0]) {
                    value[0] = attEventValue;
                }
                if (attEventValue > value[1]) {
                    value[1] = attEventValue;
                }
            });
        });

        this.data = new Instances(this.originalData);
        this.attMinMax.forEach((String attName, double[] value)->{
            Attribute numericAttribute = this.data.attribute(attName);
            Boolean isInt = false;
            List<Integer> _groupList;
            double minValue = value[0];
            double maxValue = value[1];
            if(minValue - (int)minValue == 0.0 && maxValue - (int)maxValue == 0.0){
                isInt = true;
                if((int)maxValue - (int)minValue < GROUP_NUM){
                    _groupList = new ArrayList<>(Collections.nCopies((int)maxValue - (int)minValue,0));
                } else {
                    _groupList = new ArrayList<>(Collections.nCopies(GROUP_NUM,0));
                }
            } else {
                _groupList = new ArrayList<>(Collections.nCopies(GROUP_NUM,0));
            }
            int groupNum = _groupList.size();
            value[2] = (maxValue - minValue) / groupNum;
            double split = value[2];
//            for(int i = 0, numInstance =  this.data.numInstances(); i < numInstance; i++) {
//                Instance instance =  this.data.instance(i);
            final boolean finalIsInt = isInt;
            this.data.forEach(instance -> {
                double instanceValue = instance.value(numericAttribute);
                int index = 0;
                for(; index < groupNum; index++){
                    double ceilValue;
                    if(finalIsInt){
                        ceilValue = (int)(minValue+split*index);
                    } else {
                        ceilValue = minValue+split*index;
                    }
                    if(instanceValue <= ceilValue){
                        break;
                    }
                }
                if(index == groupNum){
                    index--;
                }
                _groupList.set(index, _groupList.get(index)+1);
            });
            Tuple<List<Integer>, Boolean> groupList  = new Tuple<>(_groupList, isInt);
            this.attGroupList.put(attName, groupList);
        });

        int[] numAttributes = {this.data.numAttributes()};
        this.attGroupList.forEach((String attName, Tuple<List<Integer>, Boolean> groupList)->{
            double minValue = this.attMinMax.get(attName)[0];
            double maxValue = this.attMinMax.get(attName)[1];
            double split = this.attMinMax.get(attName)[2];
            List<Double> splitPoint = this.attSplitPoint.get(attName);
            List<Integer> _groupList = groupList.getT0();
            boolean isInt = groupList.getT1();
            int len = _groupList.size();
            int maxNo = 0;
            int maxG = _groupList.get(maxNo);
            int minG = _groupList.get(0);
            for(int i = 1; i < len; i++){
                if(_groupList.get(i) > maxG){
                    maxNo = i;
                    maxG = _groupList.get(i);
                }
            }
            boolean flag = true;
            for(int i = maxNo - 1; i > -1; i--){
                if(flag){
                    if(_groupList.get(i) < maxG * LIMIT_RATE){
                        if(isInt){
                            splitPoint.add((double)(int)(minValue+i*split+0.5));
                        }else {
                            splitPoint.add(minValue+i*split);
                        }
                        minG = _groupList.get(i);
                        flag = false;
                    }
                } else{
                    if(_groupList.get(i) > minG / LIMIT_RATE){
                        maxG = _groupList.get(i);
                        flag = true;
                    }
                }
            }
            for(int i = maxNo + 1; i < len; i++){
                if(flag){
                    if(_groupList.get(i) < maxG * LIMIT_RATE){
                        if(isInt){
                            splitPoint.add((double)(int)(minValue+i*split+0.5));
                        }else {
                            splitPoint.add(minValue+i*split);
                        }
                        minG = _groupList.get(i);
                        flag = false;
                    }
                } else{
                    if(_groupList.get(i) > minG / LIMIT_RATE){
                        maxG =_groupList.get(i);
                        flag = true;
                    }
                }
            }
            splitPoint.sort(Comparator.naturalOrder());
            Attribute numericAttribute = this.originalData.attribute(attName);
            if(splitPoint.size() == 0) { // 2分点(中位数)
                Map<Double, Integer> eventCount = getNumericEventCount(numericAttribute);
                Set<Double> sortedKeySet = new TreeSet<>(Comparator.naturalOrder());
                int cnt = 0;
                Iterator<Double> it = sortedKeySet.iterator();
                while(it.hasNext()){
                    double value = it.next();
                    cnt+=eventCount.get(value);
                    if(cnt > numAttributes[0] / 2){
                        splitPoint.add(value);
                        break;
                    }
                }
            }

            /* split the numeric data */
            List<String> attributeValues = new ArrayList<>();
            attributeValues.add("[" + integerTrimEndZero(df.format(minValue)) + "~" + integerTrimEndZero(df.format(minValue + splitPoint.get(0))) + "]");
            for(int i = 0, size = splitPoint.size()-1; i < size; i++){
                String category = "(" + integerTrimEndZero(df.format(minValue + splitPoint.get(i))) + "~" + integerTrimEndZero(df.format(minValue + splitPoint.get(i+1))) + "]";
                attributeValues.add(category);
            }
            attributeValues.add("(" + integerTrimEndZero(df.format(minValue + splitPoint.get(splitPoint.size()-1))) + "~" + integerTrimEndZero(df.format(maxValue)) + "]");
            Attribute categoryAttribute = new Attribute("_"+attName, attributeValues);
            this.data.insertAttributeAt(categoryAttribute, numAttributes[0]++);
//            for(int i = 0, numInstance = this.data.numInstances(); i < numInstance; i++) {
//                Instance instance = this.data.instance(i);
            this.data.parallelStream().forEach(instance -> {
                double value = instance.value(numericAttribute);
                int index = 0, size = splitPoint.size();
                for(; index < size; index++){
                    if(value <= splitPoint.get(index)){
                        break;
                    }
                }
                instance.setValue(numAttributes[0]-1, index);
            });
        });

        for(int i = 0; i < numAttributes[0]; i++){
            Attribute attribute = this.data.attribute(i);
            if(attribute.isNumeric()){
                this.data.deleteAttributeAt(i);
                i--;numAttributes[0]--;
            } else if(attribute.name().startsWith("_")){
                this.data.renameAttribute(i, attribute.name().substring(1));
            }
        }
        /* trim the data by selected attributes */
        this.data.setClassIndex(this.data.numAttributes() - 1);
        for(int i = 0, len_i = this.data.numAttributes(); i < len_i; i++){
            if(!this.allAttName.contains(this.data.attribute(i).name())){
                this.data.deleteAttributeAt(i);
                i--; len_i--;
            }else{
                this.data.setClassIndex(i);
            }
        }

        this.dataAggregated = new Instances(data);
    }

    private Map<Double, Integer> getNumericEventCount(Attribute numericAttribute){
        Map<Double, Integer> eventCount = new TreeMap<>();
        for (int i = 0, numInstance = this.originalData.numInstances(); i < numInstance; i++) {
            Instance instance = this.originalData.instance(i);
            double value = instance.value(numericAttribute);
            if(eventCount.containsKey(value)) {
                eventCount.put(value, eventCount.get(value)+1);
            } else{
                eventCount.put(value, 1);
            }
        }
        return eventCount;
    }

    private Instances trimData(List<String> trimList, List<JSONObject> delList) {
        Instances result = new Instances(this.data);
        IntStream.range(0, delList.size()).forEach((int idx) -> {
            JSONObject option = delList.get(idx);
            if (option == null) return;
            List<JSONObject> records = this.recommendationList.get(idx).getJSONArray("records").toJavaList(JSONObject.class);
            if (option.getBooleanValue("flag")) { // all group;
                int no = option.getIntValue("no");
                List<Integer> deletedEvents = this.recommendationList.get(idx).getJSONArray("recList").getJSONObject(no).getJSONArray("dL").toJavaList(Integer.class);

                deletedEvents.forEach((Integer eventNo) -> {
                    String deleteEventName = this.nodesMap.inverse().get(eventNo);
                    String attName = deleteEventName.split(": ")[0];
                    String value = deleteEventName.split(": ")[1];
                    Attribute att = result.attribute(attName);

                    records.forEach((JSONObject record) -> {
                        int id = record.getIntValue("id");
                        result.instance(id).setMissing(att);
                    });
                });
            } else {
                JSONArray selectionList = option.getJSONArray("selectionList");
                for (int i = 0; i < selectionList.size(); ++i) {
                    List<Integer> recordIds = selectionList.getJSONArray(i).toJavaList(Integer.class);
                    List<Integer> deletedEvents = this.recommendationList.get(idx).getJSONArray("recList").getJSONObject(i).getJSONArray("dL").toJavaList(Integer.class);

                    deletedEvents.forEach((Integer eventNo) -> {
                        String deleteEventName = this.nodesMap.inverse().get(eventNo);
                        String attName = deleteEventName.split(": ")[0];
                        String value = deleteEventName.split(": ")[1];
                        Attribute att = result.attribute(attName);

                        recordIds.forEach((Integer id) -> {
                            result.instance(id).setMissing(att);
                        });
                    });
                }
            }

        });

        for (int i = 0; i < recommendationTrimResult.size(); ++i) {
            JSONObject trimR = recommendationTrimResult.getJSONObject(i);
            String attName = trimR.getString("attributeName");

            if (this.allAttSensitivity.get(attName) || !trimList.contains(attName)) continue;

            Attribute oriAttribute = this.originalData.attribute(attName);
            Attribute resAttr = result.attribute(attName);

            if (trimR.getString("type").equals("numerical")) {
                List<JSONObject> triLi = trimR.getJSONArray("list").toJavaList(JSONObject.class);
                double min = attMinMax.get(attName)[0];
                double max = attMinMax.get(attName)[1];
                double delta = (max - min) / (triLi.size() - 1);

                for (int j = 0; j < triLi.size(); ++j) {
                    JSONObject triItem = triLi.get(j);
                    double rMin = j * delta + min;
                    double rMax = (j == triLi.size() - 1) ? max : ((j + 1) * delta + min);
                    // String value = (j == 0 ? '[' : '(') + df.format(rMin) + "~" + df.format(rMax) + "]";
                    int trimCnt = triItem.getIntValue("oriV") - triItem.getIntValue("triV");

                    List<Integer> targetIndice = new ArrayList<>();
                    for (int idx = 0; idx < this.originalData.numInstances(); ++idx) {
                        if (result.instance(idx).isMissing(resAttr)) continue;
                        double oriInsValue = this.originalData.instance(idx).value(oriAttribute);
                        if (((j == 0 && oriInsValue == rMin) || oriInsValue > rMin) && oriInsValue <= rMax) {
                            targetIndice.add(idx);
                        }
                    }
                    randomSort(targetIndice);
                    for (int cnt = 0; cnt < trimCnt && cnt < targetIndice.size(); ++cnt) {
                        result.get(targetIndice.get(cnt)).setMissing(resAttr);
                    }
                }

            } else {
                List<JSONObject> triLi = trimR.getJSONArray("data").toJavaList(JSONObject.class);
                for (int j = 0; j < triLi.size(); ++j) {
                    JSONObject triItem = triLi.get(j);
                    String cate = triItem.getString("category");
                    int trimCnt = triItem.getIntValue("oriV") - triItem.getIntValue("triV");
                    List<Integer> targetIndice = new ArrayList<>();

                    for (int idx = 0; idx < this.data.numInstances(); ++idx){
                        if (result.instance(idx).isMissing(resAttr)) continue;
                        String val = this.data.instance(idx).stringValue(resAttr.index());
                        if (cate.equals(val)) {
                            targetIndice.add(idx);
                        }
                    }

                    randomSort(targetIndice);
                    for (int cnt = 0; cnt < trimCnt && cnt < targetIndice.size(); ++cnt) {
                        result.get(targetIndice.get(cnt)).setMissing(resAttr);
                    }
                }
            }

        }

        return result;
    }

    private void randomSort(List<Integer> array) {
        for (int i = 0; i < array.size(); ++i) {
            int ri =  (int) Math.floor(Math.random() * (array.size()));
            int curVal = array.get(i);
            int rVal = array.get(ri);

            array.set(i, rVal);
            array.set(ri, curVal);
        }
    }

    private JSONObject getCorrelations(){
        int numInstances = this.data.numInstances();
        JSONObject correlations = new JSONObject();
        List<String> allNormalAtts = new ArrayList<>();
        List<String> sensitiveAtts = new ArrayList<>();
        Map<Set<String>, Tuple<Integer, List<Map<String, Integer>>>> correlationMap = new HashMap<>(); //Map(normalEventsCombination, sensitiveEvents);
        this.allAttSensitivity.forEach((attName,sensitive)->{
            if(sensitive){
                sensitiveAtts.add(attName);
            } else {
                allNormalAtts.add(attName);
            }
        });
        int numSensitiveAtts = sensitiveAtts.size();
        int numAllNormalAtts = allNormalAtts.size();
        for(int len = 1; len <= numAllNormalAtts; len++){
            for(int start = 0; start <= numAllNormalAtts - len; start++) {
                List<String> normalAtts = new ArrayList<>(allNormalAtts.subList(start, start+len-1));
                for (Instance instance : this.data) {
                    Set<String> normalEvents = new HashSet<>();
                    normalAtts.forEach(normalAttName ->
                        normalEvents.add(normalAttName + ": " + instance.stringValue(this.data.attribute(normalAttName)))
                    );
                    if (correlationMap.containsKey(normalEvents)) {
                        correlationMap.get(normalEvents).setT0(correlationMap.get(normalEvents).getT0()+1);
                    } else { // initialzation
                        List<Map<String, Integer>> sensitiveEventsList = new ArrayList<>();
                        sensitiveAtts.forEach(sensitiveAttName -> {
                            Map<String, Integer> sensitiveEvents = new HashMap<>();
                            Enumeration e = this.data.attribute(sensitiveAttName).enumerateValues();
                            while (e.hasMoreElements()) {
                                sensitiveEvents.put((String) e.nextElement(), 0);
                            }
                            sensitiveEventsList.add(sensitiveEvents);
                        });
                        correlationMap.put(normalEvents, new Tuple<>(1, sensitiveEventsList));
                    }
                    for (int i = 0; i < numSensitiveAtts; i++) {
                        String eventName = instance.stringValue(this.data.attribute(sensitiveAtts.get(i)));
                        Map<String, Integer> sensitiveAtt = correlationMap.get(normalEvents).getT1().get(i);
                        sensitiveAtt.put(eventName, sensitiveAtt.get(eventName) + 1);
                    }
                }
            }
        }
        for (int i = 0; i < numSensitiveAtts; i++) {
            String sensitiveAttName = sensitiveAtts.get(i);
            Enumeration e = this.data.attribute(sensitiveAttName).enumerateValues();
            while (e.hasMoreElements()) {
                String eventName = (String)e.nextElement();
                String sensitiveEventName = sensitiveAttName+": "+eventName;
                JSONObject correlation = new JSONObject();
                List<JSONObject> probabilityList = new ArrayList<>();
                final int index = i;
                correlationMap.forEach((normalEventSet, tuple)->{
                    double correlationValue = (double) tuple.getT1().get(index).get(eventName) / tuple.getT0();
                    JSONObject probabilities = new JSONObject();
                    probabilities.put("eventLists", normalEventSet);
                    probabilities.put("cor", correlationValue);
                    probabilities.put("freq", tuple.getT0());
                    probabilityList.add(probabilities);
                });
                probabilityList.sort((o1, o2) -> {
                    double v1 = o1.getDoubleValue("cor");
                    double v2 = o2.getDoubleValue("cor");
                    if(v1 > v2){
                        return -1;
                    } else if(v1 < v2){
                        return 1;
                    } else{
                        return 0;
                    }
                });
                correlation.put("data", probabilityList);
                correlation.put("pro", (double) this.priorMap.get(sensitiveEventName) / numInstances);
                correlations.put(sensitiveEventName, correlation);
            }
        }
        return correlations;
    }

    /**
     * used for function test
     * @param args
     */
    public static void main(String[] args) {
//        Bayes bn = new Bayes("home");
//        List<DataSegment> test= new ArrayList<>();
//        test.add(new DataSegment(4, 5));
//        test.add(new DataSegment(2, 3));
//        test.contains(new DataSegment(2,4));
//        test.contains(new DataSegment(3,5));
//
//        String li = "[{\"attName\":\"fmp\",\"sensitive\":false},{\"attName\":\"emp\",\"sensitive\":true},{\"attName\":\"gen\",\"sensitive\":false},{\"attName\":\"gcs\",\"sensitive\":false},{\"attName\":\"cat\",\"sensitive\":false},{\"attName\":\"fue\",\"sensitive\":false},{\"attName\":\"sch\",\"sensitive\":false}]";
//        List<JSONObject> arr = JSON.parseArray(li, JSONObject.class);
//
//        bn.getGBN(arr);
        // bn.getTest("KNN", null);

    }
}

