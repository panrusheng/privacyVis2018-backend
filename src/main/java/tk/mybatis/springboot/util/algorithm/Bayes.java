package tk.mybatis.springboot.util.algorithm;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import eu.amidst.core.datastream.*;
import eu.amidst.core.datastream.filereaders.DataStreamFromFile;
import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.ParameterLearningAlgorithm;
import tk.mybatis.springboot.util.MyArffReader;

//import java.io.*;

//import java.nio.file.Files;
//import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;

public class Bayes {
    private static String root_path = "C:\\Users\\Echizen\\Documents\\work\\privacyVis2018\\src\\main\\java\\";
    public static String getGBN(){
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();
        DecimalFormat df = new DecimalFormat("#0.00");//To use: (String) df.format(Number);
        MyArffReader myArffReader = new MyArffReader();

        myArffReader.loadFromFile(root_path+"tk\\mybatis\\springboot\\data\\user.arff");
        DataStream<DataInstance> dataStream = new DataStreamFromFile(myArffReader);
        Attributes attributes = dataStream.getAttributes();
        Variables modelHeader = new Variables(attributes);
        DAG dag = new DAG(modelHeader);
        int[] attList = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
        JSONObject gbn = new JSONObject();
        JSONArray nodesList = new JSONArray();
        JSONArray linksList = new JSONArray();
        Map<String, Double[]> probTable = new HashMap<>();
        Map<String, Integer> EventNo = new HashMap<>();//Key: attName, Value: first event no.
        int totalEventNo = 0;
        for(int i = 0, len_i = attList.length; i < len_i; i++){
            /* get probability table */
            parameterLearningAlgorithm.setDAG(dag);
            parameterLearningAlgorithm.initLearning();
            parameterLearningAlgorithm.updateModel(dataStream);
            BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

            int attNo = attList[i];
            Variable attNode = modelHeader.getVariableById(attNo);
            String attName = attNode.getName();
            Attribute att = attributes.getAttributeByName(attName);

            String __probDistList = bnModel.getConditionalDistribution(attNode).toString();
            String[] _probDistList = __probDistList.substring(2, __probDistList.length()-2).split(", ");
            Double[] probDistList = new Double[_probDistList.length];
            EventNo.put(attName, totalEventNo);
            double min = myArffReader.min[i];
            double max = myArffReader.max[i];
            String mid = df.format((min + max) / 2);
            for(int j = 0, len_j = _probDistList.length; j < len_j; j++){
                probDistList[j] = Double.valueOf(_probDistList[j]);

                /* generate the node of eventNodeList */
                JSONObject node = new JSONObject();
                String _eventName = att.stringValue((double)j);
                String eventName;
                if(_eventName.equals("category_0")){
                    eventName = "[" + min + "-" + mid + ")";
                }
                else if(_eventName.equals("category_1")){
                    eventName = "[" + mid + "-" + max + "]";
                }
                else{
                    eventName = "_" + _eventName;
                }
                node.put("id", attName + eventName);
                node.put("attName", attName);
                node.put("eventNo", totalEventNo++);
                node.put("value", 1);
                nodesList.add(node);
            }
            probTable.put(attName, probDistList);
        }

        for(int i : attList){//parent node
            Variable parentVar = modelHeader.getVariableById(i);
            for(int j : attList){//child node
                if(j != i){
                    Variable childVar = modelHeader.getVariableById(j);
                    String childVarName = childVar.getName();
                    String parentVarName = parentVar.getName();
                    /* add single parent */
                    dag.getParentSet(childVar).addParent(parentVar);

                    parameterLearningAlgorithm.setDAG(dag);
                    parameterLearningAlgorithm.initLearning();
                    parameterLearningAlgorithm.updateModel(dataStream);

                    BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();
                    ConditionalDistribution condDist = bnModel.getConditionalDistribution(childVar);
                    String[]  condDistList = condDist.toString().split("\n");
                    for(int k = 0, len_k = condDistList.length; k < len_k; k++){
                        String[] probDistList_and_condition = condDistList[k].split("\\|");
                        String __probDistList = probDistList_and_condition[0].trim();
                        String[] _probDistList = __probDistList.substring(2, __probDistList.length()-2).split(", ");
                        for( int m = 0, len_m = _probDistList.length; m < len_m; m++ ){
                            String[] cpt = new String[4]; //cpt = [ P(A), P(B), P(A|B), P(A|B') ]
                            cpt[0] = df.format(probTable.get(childVarName)[m]); //P(A)
                            cpt[1] = df.format(probTable.get(parentVarName)[k]); //P(B)
                            cpt[2] = df.format(Double.valueOf(_probDistList[m])); //P(A|B)
                            int cpt3_n = 0, cpt3_d = 0;
                            for(DataInstance record : dataStream){
                                Attributes recordAtt = record.getAttributes();
                                int childAtt = (int)record.getValue(recordAtt.getAttributeByName(childVarName));
                                int parentAtt = (int)record.getValue(recordAtt.getAttributeByName(parentVarName));
                                if(parentAtt != k){
                                    cpt3_d++;
                                    if(childAtt == m){
                                        cpt3_n++;
                                    }
                                }
                            }
                            cpt[3] = df.format((double)cpt3_n / (double)cpt3_d);
                            JSONObject link = new JSONObject();
                            link.put("source", EventNo.get(parentVarName)+k);
                            link.put("target", EventNo.get(childVarName)+m);
                            link.put("value", cpt[2]);
                            link.put("cpt", cpt);
                            linksList.add(link);
                        }
                    }
                    /* remove the parent */
                    dag.getParentSet(childVar).removeParent(parentVar);
                }
            }
        }
        gbn.put("nodes",nodesList);
        gbn.put("links",linksList);
        return gbn.toJSONString();
    }
}

