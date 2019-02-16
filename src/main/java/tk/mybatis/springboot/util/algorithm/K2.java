package tk.mybatis.springboot.util.algorithm;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataStream;
import tk.mybatis.springboot.util.MyMath;

import java.util.HashSet;
import java.util.Set;

public class K2 {
    private DataStream<DataInstance> dataStream;
    private double f(int i, Set pi_i){
        double result = 1.0;
        int q_i = 10, r_i = 10;
        for(int j = 1; j < q_i; j++){
            result *= MyMath.fact(r_i - 1, 1);
        }
        return result;
    }
    public int[][] construct(DataStream<DataInstance> dataStream, int[] attList, int parentUpperBound){
        this.dataStream = dataStream;
        int attListLength = attList.length;
        int[][] parList = new int[attListLength][];
        Set[] parSet = new HashSet[attListLength];
        for (int i = 0, len_i = attListLength; i < len_i; i++){
            parSet[i] = new HashSet<Integer>();
            double p_old = f(i,parSet[i]);
            boolean okToProceed = true;
            while(okToProceed && parSet[i].size() < parentUpperBound){
                int z = 10;//test
                okToProceed = false;
            }
        }
        return parList;
    }
}
