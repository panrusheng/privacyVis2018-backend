package tk.mybatis.springboot.util.algorithm;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataStream;
import eu.amidst.core.io.BayesianNetworkWriter;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.learning.parametric.ParallelMaximumLikelihood;
import eu.amidst.core.learning.parametric.ParameterLearningAlgorithm;
import eu.amidst.dynamic.models.DynamicBayesianNetwork;
import eu.amidst.dynamic.utils.DynamicBayesianNetworkGenerator;

public class Bayes {
    private static DAG getNaiveBayesStructure(DataStream<DataInstance> dataStream){

        //We create a Variables object from the attributes of the data stream
        Variables modelHeader = new Variables(dataStream.getAttributes());

        Variable id = modelHeader.getVariableByName("id");
        Variable wei = modelHeader.getVariableByName("wei");
        Variable gen = modelHeader.getVariableByName("gen");
        Variable cat = modelHeader.getVariableByName("cat");
        Variable res = modelHeader.getVariableByName("res");
        Variable sch = modelHeader.getVariableByName("sch");
        Variable fue = modelHeader.getVariableByName("fue");
        Variable gcs = modelHeader.getVariableByName("gcs");
        Variable fmp = modelHeader.getVariableByName("fmp");
        Variable lvb = modelHeader.getVariableByName("lvb");
        Variable tra = modelHeader.getVariableByName("tra");
        Variable emp = modelHeader.getVariableByName("emp");
        Variable jol = modelHeader.getVariableByName("jol");
        Variable fe = modelHeader.getVariableByName("fe");
        Variable he = modelHeader.getVariableByName("he");
        Variable ascc = modelHeader.getVariableByName("ascc");

        //Then, we create a DAG object with the defined model header
        DAG dag = new DAG(modelHeader);

//        dag.getParentSet(sch).addParent(res);
//        dag.getParentSet(lvb).addParent(sch);
        int[] attrList = {2,3,4,5,6,7,8,9};
        for(int i : attrList){
            Variable classVar = modelHeader.getVariableById(i);
            dag.getParentSets().stream().filter(w -> w.getMainVar() != classVar).forEach(w -> w.addParent(classVar));
        }
        System.out.println(dag.toString());
        return dag;
    }
    public static void test() {
        //We can open the data stream using the static class DataStreamLoader
        DataStream<DataInstance> data = DataStreamLoader.open("C:\\Users\\Echizen\\Documents\\work\\privacyVis2018\\src\\main\\java\\tk\\mybatis\\springboot\\data\\user.arff");

        //We create a ParameterLearningAlgorithm object with the MaximumLikehood builder
        ParameterLearningAlgorithm parameterLearningAlgorithm = new ParallelMaximumLikelihood();

        //We fix the DAG structure
        parameterLearningAlgorithm.setDAG(getNaiveBayesStructure(data));

        //We should invoke this method before processing any data
        parameterLearningAlgorithm.initLearning();


        //Then we show how we can perform parameter learning by a sequential updating of data batches.
        for (DataOnMemory<DataInstance> batch : data.iterableOverBatches(100)){
            parameterLearningAlgorithm.updateModel(batch);
        }

        //And we get the model
        BayesianNetwork bnModel = parameterLearningAlgorithm.getLearntBayesianNetwork();

        //We print the model
        System.out.println(bnModel.toString());
    }

}
