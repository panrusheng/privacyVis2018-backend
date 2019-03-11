package tk.mybatis.springboot.util.algorithm;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.local.*;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.neighboursearch.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

enum ClassifierModel {
    SVM, RandomForest, BayesianNetwork, KNN, DecisionTree
}

enum NeighborSearchAlgorithm {
    BallTree, CoverTree, FilteredNeighourSearch, KDTree, LinearNNSearch,
}

enum Distance {
    EuclideanDistance, FilteredDistance, ManhattanDistance, MinkowskiDistance, ChebyshevDistance
}

public class Model {
    private static void setOptions(Classifier model, JSONObject options) throws Exception {
        if (options == null) return;

        if (model instanceof LibSVM) {
            if (options.getInteger("kernelType") != null) {
                ((LibSVM)model).setKernelType(new SelectedTag(options.getIntValue("kernelType"), LibSVM.TAGS_KERNELTYPE));

            }
            if (options.getInteger("degree") != null) {
                ((LibSVM)model).setDegree(options.getIntValue("degree"));
            }
            if (options.getDouble("gamma") != null) {
                ((LibSVM)model).setGamma(options.getDoubleValue("gamma"));
            }
            if (options.getDouble("coef0") != null) {
                ((LibSVM)model).setCoef0(options.getDoubleValue("coef0"));
            }
        } else if (model instanceof BayesNet) {
            if (options.getString("searchAlgorithm") != null) {
                switch (LocalSearchAlgorithm.valueOf(options.getString("searchAlgorithm"))) {
                    case K2:{
                        K2 algorithm = new K2();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    }break;
                    case GeneticSearch:{
                        GeneticSearch algorithm = new GeneticSearch();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    }
                    case HillClimber:{
                        HillClimber algorithm = new HillClimber();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    case LAGDHillClimber:{
                        LAGDHillClimber algorithm = new LAGDHillClimber();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    case LocalScoreSearchAlgorithm:{
                        LocalScoreSearchAlgorithm algorithm = new LocalScoreSearchAlgorithm();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    case RepeatedHillClimber:{
                        RepeatedHillClimber algorithm = new RepeatedHillClimber();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    case SimulatedAnnealing:{
                        SimulatedAnnealing algorithm = new SimulatedAnnealing();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    case TabuSearch:{
                        TabuSearch algorithm = new TabuSearch();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    case TAN:{
                        TAN algorithm = new TAN();
                        ((BayesNet)model).setSearchAlgorithm(algorithm);
                    } break;
                    default:break;
                }
            }
        } else if (model instanceof RandomForest) {
            if (options.getInteger("maxDepth") != null) {
                ((RandomForest)model).setMaxDepth(options.getIntValue("maxDepth"));
            }
            if (options.getString("batchSize") != null) {
                ((RandomForest)model).setBatchSize(options.getString("batchSize"));
            }
        } else if (model instanceof IBk) {
            if (options.containsKey("searchAlgorithm")) {
                NearestNeighbourSearch algorithm = null;
                switch (NeighborSearchAlgorithm.valueOf(options.getString("searchAlgorithm"))) {
                    case KDTree: algorithm = new KDTree(); break;
                    case BallTree: algorithm = new BallTree(); break;
                    case CoverTree: algorithm = new CoverTree(); break;
                    case LinearNNSearch: algorithm = new LinearNNSearch(); break;
                    case FilteredNeighourSearch: algorithm = new FilteredNeighbourSearch(); break;
                }

                if (options.containsKey("distanceFunction")) {
                    DistanceFunction df = null;
                    switch (Distance.valueOf(options.getString("distanceFunction"))) {
                        case FilteredDistance: df = new FilteredDistance(); break;
                        case ChebyshevDistance: df = new ChebyshevDistance(); break;
                        case EuclideanDistance: df = new EuclideanDistance(); break;
                        case ManhattanDistance: df = new ManhattanDistance(); break;
                        case MinkowskiDistance: df = new MinkowskiDistance(); break;
                    }

                    algorithm.setDistanceFunction(df);
                }

                ((IBk)model).setNearestNeighbourSearchAlgorithm(algorithm);
            }

            if (options.containsKey("k")) {
                ((IBk)model).setKNN(options.getIntValue("k"));
            }
            if (options.containsKey("distanceWeighting")) {
                ((IBk)model).setDistanceWeighting(new SelectedTag(options.getIntValue("distanceWeighting"), IBk.TAGS_WEIGHTING));
            }
            if (options.containsKey("crossValidation")) {
                ((IBk)model).setCrossValidate(options.getBooleanValue("crossValidation"));
            }

        } else if (model instanceof J48) {
            if (options.containsKey("unpruned")) {
                ((J48)model).setUnpruned(options.getBooleanValue("unpruned"));
            }
            if (options.containsKey("confidenceThreshold")) {
                ((J48)model).setConfidenceFactor(options.getFloatValue("confidenceThreshold"));
            }
            if (options.containsKey("minInstance")) {
                ((J48)model).setMinNumObj(options.getIntValue("minInstance"));
            }
            if (options.containsKey("laplaceSmoothing")) {
                ((J48)model).setUseLaplace(options.getBooleanValue("laplaceSmoothing"));
            }
            if (options.containsKey("reducedErrorPruning")) {
                ((J48)model).setReducedErrorPruning(options.getBooleanValue("reducedErrorPruning"));
            }
            if (options.containsKey("MDLCorrection")) {
                ((J48)model).setUseMDLcorrection(options.getBooleanValue("MDLCorrection"));
            }
            if (options.containsKey("collapseTree")) {
                ((J48)model).setCollapseTree(options.getBooleanValue("collapseTree"));
            }
            if (options.containsKey("subtreeRaising")) {
                ((J48)model).setSubtreeRaising(options.getBooleanValue("subtreeRaising"));
            }

        }
    }

    private static Classifier getModel(String classifier) {
        switch (ClassifierModel.valueOf(classifier)) {
            case SVM: return new LibSVM();
            case BayesianNetwork: return new BayesNet();
            case RandomForest: return new RandomForest();
            case KNN: return new IBk();
            case DecisionTree: return new J48();
            default: return null;
        }
    }

    public static JSONArray test(String classifier, Instances oriD, Instances proD, JSONObject options) throws Exception {
        Classifier modelOriD = getModel(classifier);
        Classifier modelProD = getModel(classifier);
        setOptions(modelOriD, options);
        setOptions(modelProD, options);

        JSONArray list = new JSONArray();

        modelOriD.buildClassifier(oriD);
        modelProD.buildClassifier(proD);

        Attribute classAtt = oriD.classAttribute();
        String attName = classAtt.name();
        int numVals = classAtt.numValues();

        for (int i = 0; i < numVals; ++i) {
            JSONObject event = new JSONObject();
            String value = classAtt.value(i);
            String eventName = attName + ": " + value;

            int frequency = 0;
            int tp, tn, fp, fn;
            tp = tn = fp = fn = 0;

            for (Instance instance : oriD) {
                String predictVal = classAtt.value((int) modelOriD.classifyInstance(instance));
                String realVal = instance.stringValue(classAtt);

                if (realVal.equals(value)) {
                    if (predictVal.equals(value)) {
                        tp++;
                    } else {
                        fn++;
                    }
                } else {
                    if (!predictVal.equals(value)) {
                        tn++;
                    } else {
                        fp++;
                    }
                }
                if (realVal.equals(value)) {
                    frequency++;
                }
            }

            JSONObject measure = new JSONObject();
            measure.put("TP", tp);
            measure.put("TN", tn);
            measure.put("FP", fp);
            measure.put("FN", fn);
            measure.put("sensitivity", (double)tp / (tp + fn));
            measure.put("specificity", (double)tn / (tn + fp));

            event.put("oriD", measure);

            tp = tn = fp = fn = 0;

            for (Instance instance : proD) {
                String predictVal = classAtt.value((int) modelProD.classifyInstance(instance));
                String realVal = instance.stringValue(classAtt);

                if (realVal.equals(value)) {
                    if (predictVal.equals(value)) {
                        tp++;
                    } else {
                        fn++;
                    }
                } else {
                    if (!predictVal.equals(value)) {
                        tn++;
                    } else {
                        fp++;
                    }
                }
            }

            measure = new JSONObject();
            measure.put("TP", tp);
            measure.put("TN", tn);
            measure.put("FP", fp);
            measure.put("FN", fn);
            measure.put("sensitivity", (double)tp / (tp + fn));
            measure.put("specificity", (double)tn / (tn + fp));

            event.put("proD", measure);
            event.put("eveName", eventName);
            event.put("frequency", frequency);
            list.add(event);
            // System.out.printf("%s %d, %d, %d, %d\n", eventName, tp, tn, fp, fn);
        }

        return list;
    }

    public static void main(String[] args) throws Exception {
        String root_path = new File(".").getAbsoluteFile().getParent()
                + File.separator + "src"+ File.separator + "main"+ File.separator + "java"+ File.separator;
        BufferedReader reader = new BufferedReader(new FileReader(root_path + "tk\\mybatis\\springboot\\data\\user.arff"));
        Instances instances = new Instances(reader);
        Discretize discretize = new Discretize();
        discretize.setInputFormat(instances);
        instances = Filter.useFilter(instances, discretize);
        instances.setClassIndex(instances.numAttributes() - 1);
        JSONObject options = new JSONObject();
        options.put("distanceWeighting", 1);
        for (Tag t : IBk.TAGS_WEIGHTING) {
            System.out.println(t.getReadable());
        }

        System.out.println(Model.test("KNN", instances, instances, options));
    }
}
