package tk.mybatis.springboot.util.algorithm;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import eu.amidst.core.datastream.*;
import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.io.BayesianNetworkWriter;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Assignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.ParameterLearningAlgorithm;

import java.io.*;

import java.text.DecimalFormat;
import java.util.*;

public class Bayes {
    private static String root_path = "C:\\Users\\Echizen\\Documents\\work\\privacyVis2018\\src\\main\\java\\";
//    private static DAG getNaiveBayesStructure(DataStream<DataInstance> dataStream){
//
//        //We create a Variables object from the attributes of the data stream
//        Variables modelHeader = new Variables(dataStream.getAttributes());
//
//        Variable id = modelHeader.getVariableByName("id");
//        Variable wei = modelHeader.getVariableByName("wei");
//        Variable gen = modelHeader.getVariableByName("gen");
//        Variable cat = modelHeader.getVariableByName("cat");
//        Variable res = modelHeader.getVariableByName("res");
//        Variable sch = modelHeader.getVariableByName("sch");
//        Variable fue = modelHeader.getVariableByName("fue");
//        Variable gcs = modelHeader.getVariableByName("gcs");
//        Variable fmp = modelHeader.getVariableByName("fmp");
//        Variable lvb = modelHeader.getVariableByName("lvb");
//        Variable tra = modelHeader.getVariableByName("tra");
//        Variable emp = modelHeader.getVariableByName("emp");
//        Variable jol = modelHeader.getVariableByName("jol");
//        Variable fe = modelHeader.getVariableByName("fe");
//        Variable he = modelHeader.getVariableByName("he");
//        Variable ascc = modelHeader.getVariableByName("ascc");
//
//        //Then, we create a DAG object with the defined model header
//        DAG dag = new DAG(modelHeader);
//
////        dag.getParentSet(sch).addParent(res); // sch <- res
////        dag.getParentSet(lvb).addParent(sch); // lvb <- sch
//        int[] attrList = {2,3,4,5,6,7,8,9};
//        for(int i : attrList){
//            Variable classVar = modelHeader.getVariableById(i);
////            dag.getParentSets().stream().filter(w -> w.getMainVar() != classVar).forEach(w -> w.addParent(classVar));
//            for(int j : attrList){
//                if(j != i){
//                    dag.getParentSet(modelHeader.getVariableById(j)).addParent(classVar);
//                }
//            }
//        }
////        System.out.println(dag.getParentSet(sch).getMainVar().getName());
//        System.out.println(dag.toString());
//        return dag;
//    }
//    public static void test() {
//        //We can open the data stream using the static class DataStreamLoader
//        DataStream<DataInstance> data = DataStreamLoader.open(root_path+"tk\\mybatis\\springboot\\data\\user.arff");
//
//        //We create a ParameterLearningAlgorithm object with the MaximumLikehood builder
//        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();
//
//        //We fix the DAG structure
//        parameterLearningAlgorithm.setDAG(getNaiveBayesStructure(data));
//
//        //We should invoke this method before processing any data
//        parameterLearningAlgorithm.initLearning();
//
//        //Then we show how we can perform parameter learning by a sequential updating of data batches.
//        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(300)){
//            parameterLearningAlgorithm.updateModel(batch);
//        }
//
//        //And we get the model
//        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();
//
//        JSONObject gbn = new JSONObject();
//
//        // 保存数据文件
//
//        // 拼接文件完整路径
//        String fullPath = root_path+"tk\\mybatis\\springboot\\data\\" + File.separator + "result.txt";
//
//        // 生成json格式文件
//        try {
//            // 保证创建一个新文件
//            File file = new File(fullPath);
//            if (!file.getParentFile().exists()) { // 如果父目录不存在，创建父目录
//                file.getParentFile().mkdirs();
//            }
//            if (file.exists()) { // 如果已存在,删除旧文件
//                file.delete();
//            }
//            file.createNewFile();
//
//            Writer write = new OutputStreamWriter(new FileOutputStream(file), "UTF-8");
//            write.write(bnModel.toString());
//            write.flush();
//            write.close();
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//    }
    public static String getGBN(){

        DecimalFormat df = new DecimalFormat("#0.00");
        DataStream<DataInstance> data = DataStreamLoader.open(root_path+"tk\\mybatis\\springboot\\data\\user.arff");
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();

        Variables modelHeader = new Variables(data.getAttributes());
        DAG dag = new DAG(modelHeader);

        int[] attrList = {2,3,4,5,6,7,8,9};
        JSONObject gbn = new JSONObject();
        JSONArray nodesList = new JSONArray();
        JSONArray linksList = new JSONArray();
        Map<String, Double[]> probTable = new HashMap<>();
        Map<String, Integer> EventNo = new HashMap<>();//Key: attrName, Value: first event no.
        int totalEventNo = 0;

        for(int i = 0, len_i = attrList.length; i < len_i; i++){

            /* get probability table */
            parameterLearningAlgorithm.setDAG(dag);
            parameterLearningAlgorithm.initLearning();
            parameterLearningAlgorithm.updateModel(data);
            BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

            int attrNo = attrList[i];
            Variable attrNode = modelHeader.getVariableById(attrNo);
            String attrName = attrNode.getName();
            String __probDistList = bnModel.getConditionalDistribution(attrNode).toString();
            String[] _probDistList = __probDistList.substring(2, __probDistList.length()-2).split(", ");
//            System.out.println(attrNode.getName());
//            System.out.println(probDist);

            Double[] probDistList = new Double[_probDistList.length];
            for(int j = 0, len_j = _probDistList.length; j < len_j; j++){
                probDistList[j] = Double.valueOf(_probDistList[j]);

                /* generate the node of eventNodeList */
                JSONObject node = new JSONObject();
                node.put("id", attrName+"_"+String.valueOf(j));
                node.put("attrName", attrName);
                node.put("eventNo", totalEventNo++);
                node.put("value", 1);
                nodesList.add(node);
            }
            probTable.put(attrName, probDistList);
            EventNo.put(attrName, totalEventNo);

        }
//        System.out.println(probTable.get("lvb")[0]);

        for(int i : attrList){//parent node
            Variable parentVar = modelHeader.getVariableById(i);
            for(int j : attrList){//child node
                if(j != i){
                    Variable childVar = modelHeader.getVariableById(j);
                    String childVarName = childVar.getName();
                    String parentVarName = parentVar.getName();
                    /* add single parent */
                    dag.getParentSet(childVar).addParent(parentVar);

                    parameterLearningAlgorithm.setDAG(dag);
                    parameterLearningAlgorithm.initLearning();
                    parameterLearningAlgorithm.updateModel(data);

                    BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

                    /* get child node's name */
//                    System.out.println(childVar.getName());

                    ConditionalDistribution condDist = bnModel.getConditionalDistribution(childVar);
                    String[]  condDistList = condDist.toString().split("\n");
                    for(int k = 0, len_k = condDistList.length; k < len_k; k++){
                        String[] probDistList_and_condition = condDistList[k].split("\\|");
                        String __probDistList = probDistList_and_condition[0].trim();
//                        String _condition = probDistList_and_condition[1].trim();
                        String[] _probDistList = __probDistList.substring(2, __probDistList.length()-2).split(", ");
//                        String condition = _condition.substring(1, _condition.length()-1);
                        for( int m = 0, len_m = _probDistList.length; m < len_m; m++ ){
                            Double[] cpt = new Double[4]; //cpt = [ P(A), P(B), P(A|B), P(A|B') ]
                            cpt[0] = probTable.get(childVarName)[m]; //P(A)
                            cpt[1] = probTable.get(parentVarName)[k]; //P(B)
                            cpt[2] = Double.valueOf(_probDistList[m]); //P(A|B)
                            cpt[3] = 0.0; //P(A|B'), placeholder
                            double cpt3_n = 0.0, cpt3_d = 0.0;
                            for(Iterator<DataInstance> iter = data.iterator();iter.hasNext();){
                                DataInstance record = iter.next();
                                Attributes recordAttr = record.getAttributes();
                                int childAttr = (int)record.getValue(recordAttr.getAttributeByName(childVarName));
                                int parentAttr = (int)record.getValue(recordAttr.getAttributeByName(parentVarName));
                                if(parentAttr != k){
                                    cpt3_d++;
                                    if(childAttr == m){
                                        cpt3_n++;
                                    }
                                }
                            }
                            cpt[3] = cpt3_n / cpt3_d;

                            JSONObject link = new JSONObject();
                            link.put("source", EventNo.get(parentVarName)+k);
                            link.put("target", EventNo.get(childVarName)+m);
                            link.put("value", _probDistList[m]);
                            link.put("cpt", cpt);
                            linksList.add(link);
                        }

                    }
//                    System.out.println(data.getAttributes());
                    /* get parent(condition) node's name */
//                    System.out.println(condDistr.getConditioningVariables().get(0).getName());

                    /* remove the parent */
                    dag.getParentSet(childVar).removeParent(parentVar);
                }
            }
        }
//        System.out.println(dag.getParentSet(sch).getMainVar().getName());
//        System.out.println(dag.toString());
        System.out.println(nodesList);
        System.out.println(linksList);
        gbn.put("nodes",nodesList);
        gbn.put("links",linksList);

        return gbn.toJSONString();

    }

}

