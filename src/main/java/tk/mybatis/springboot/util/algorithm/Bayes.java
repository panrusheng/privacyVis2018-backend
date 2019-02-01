package tk.mybatis.springboot.util.algorithm;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

//import java.io.*;

//import java.nio.file.Files;
//import java.nio.file.Paths;
import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.*;

public class Bayes {
    public static String getGBN() {
        JSONObject gbn = new JSONObject();
        JSONArray nodeList = new JSONArray();
        JSONArray linkList = new JSONArray();

        try {
            BufferedReader reader = new BufferedReader(new FileReader("C:\\Users\\kotta\\code\\projects\\privacyVis2018-backend\\src\\main\\java\\tk\\mybatis\\springboot\\data\\user.arff"));
            Instances originalData = new Instances(reader);

            Discretize discretize = new Discretize();
            discretize.setBins(2);
            discretize.setInputFormat(originalData);

            Instances data = Filter.useFilter(originalData, discretize);
            data.setClassIndex(data.numAttributes() - 1);

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
                    String value = bn.getNodeValue(i, j);
                    node.put("attrName", attr);
                    node.put("id", attr + "_" + value);
                    node.put("value", 1);
                    node.put("eventNo", incrId);
                    nodeList.add(node);
                    eventNoMap.put(attr + "_" + value, incrId++);
                }
            }

            for (int i = 0; i < data.numAttributes(); ++i) {
                for (int j = 0; j < data.numInstances(); ++j) {
                    String id = data.attribute(i).name() + "_" + data.instance(j).stringValue(i);
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
                    String val = bn.getNodeValue(iNode, iValue);
                    String childId = attr + "_" + bn.getNodeValue(iNode, iValue);

                    for (int iParent = 0; iParent < bn.getNrOfParents(iNode); ++iParent) {
                        int parent = bn.getParent(iNode, iParent);
                        String attrParent = bn.getNodeName(parent);
                        for (int m = 0; m < bn.getCardinality(parent); ++m) {
                            String valParent = bn.getNodeValue(parent, m);
                            String parentId = attrParent + "_" + bn.getNodeValue(parent, m);
                            JSONObject link = new JSONObject();
                            double[] cpt = new double[4];
                            cpt[0] = (double)priorMap.get(childId) / numInstances;
                            cpt[1] = (double)priorMap.get(parentId) / numInstances;

                            int cpt2_p = 0, cpt2_c = 0, cpt3_p = 0, cpt3_c = 0;

                            for (int a = 0; a < data.numInstances(); ++a) {
                                Instance instance = data.instance(a);

                                if (instance.stringValue(data.attribute(attrParent)).equals(valParent)) {
                                    cpt2_p++;
                                    if (instance.stringValue(data.attribute(attr)).equals(val)) {
                                        cpt2_c++;
                                    }
                                } else {
                                    cpt3_p++;
                                    if (instance.stringValue(data.attribute(attr)).equals(val)) {
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

    public static void main(String[] args) {
        Bayes.getGBN();
    }

}

