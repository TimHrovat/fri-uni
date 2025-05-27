import java.util.*;
import java.util.function.IntBinaryOperator;

public class Postevanka {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();

        // postevanka(n, new Krat());

        postevanka(n, (a, b) -> a * b);
    }

    private static class Plus implements IntBinaryOperator {
        @Override
        public int applyAsInt(int a, int b) {
            return a + b; 
        }
    }

    private static class Krat implements IntBinaryOperator {
        @Override
        public int applyAsInt(int a, int b) {
            return a * b; 
        }
    }

    private static void postevanka(int n, IntBinaryOperator operacija) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                System.out.printf("%4d", operacija.applyAsInt(i, j));
            }

            System.out.println();
        }
    }
}
