import java.util.Scanner;
import java.lang.Math;

public class DN03_63230112 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int h = sc.nextInt();
        int w = sc.nextInt();
        int k = sc.nextInt(); // max size ... 2^n
        sc.close();

        long sum = 0;
        long[] squares = new long[k + 1];

        for (int i = 0; i <= k; i++) {
            long tiles_w = w / (long) Math.pow(2, k - i);
            long tiles_h = h / (long) Math.pow(2, k - i);

            squares[i] = tiles_w * tiles_h;

            if (i == 0) {
                sum += tiles_w * tiles_h;
                continue;
            }

            sum += tiles_w * tiles_h - squares[i - 1] * 4;
        }

        System.out.println(sum);
    }
}