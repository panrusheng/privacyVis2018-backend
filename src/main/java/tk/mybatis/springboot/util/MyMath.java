package tk.mybatis.springboot.util;

public class MyMath {
    public static long fact(long n){
        return fact(n, 1);
    }

    private static long fact(long n, long lastValue){
        if(n < 0) return 0;
        else if( n == 0) return 1;
        else if( n == 1) return lastValue;
        return fact(n-1, n*lastValue);
    }
}
