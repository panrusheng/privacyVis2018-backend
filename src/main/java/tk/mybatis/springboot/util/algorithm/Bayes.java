package tk.mybatis.springboot.util.algorithm;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

        DataStream<DataInstance> data = DataStreamLoader.open(root_path+"tk\\mybatis\\springboot\\data\\user.arff");
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();
        Variables modelHeader = new Variables(data.getAttributes());
        DAG dag = new DAG(modelHeader);

        int[] attrList = {2,3,4,5,6,7,8,9};
        JSONObject gbn = new JSONObject();
        JSONArray nodesList = new JSONArray();
        JSONArray linksList = new JSONArray();
        for(int i : attrList){
            Variable parentVar = modelHeader.getVariableById(i);

            JSONObject node = new JSONObject();
            node.put("id", i);
            node.put("attrName", parentVar.getName());
            node.put("value", 1);
            nodesList.add(node);
            for(int j : attrList){
                if(j != i){
                    Variable childVar = modelHeader.getVariableById(j);
                    //add single parent
                    dag.getParentSet(childVar).addParent(parentVar);

                    parameterLearningAlgorithm.setDAG(dag);
                    parameterLearningAlgorithm.initLearning();
                    parameterLearningAlgorithm.updateModel(data);

                    BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

                    /* get child node's name */
//                    System.out.println(childVar.getName());

                    ConditionalDistribution condDistr = bnModel.getConditionalDistribution(childVar);
//                    String[]  condDistrList = condDistr.toString().split("\n");
//                    for( String distribution: condDistrList){
//                        System.out.println(distribution);
//                    }
//                    System.out.println(data.getAttributes());
                    /* get parent(condition) node's name */
//                    System.out.println(condDistr.getConditioningVariables().get(0).getName());

                    JSONObject link = new JSONObject();
                    link.put("source",condDistr.getConditioningVariables().get(0).getName());
                    link.put("target",childVar.getName());
                    link.put("value",condDistr.toString());
                    linksList.add(link);
                    /* remove the parent */
                    dag.getParentSet(childVar).removeParent(parentVar);
                }
            }
        }
//        System.out.println(dag.getParentSet(sch).getMainVar().getName());
//        System.out.println(dag.toString());
        System.out.println(nodesList);
        gbn.put("nodes",nodesList);
        gbn.put("links",linksList);

        return gbn.toJSONString();

    }

}

