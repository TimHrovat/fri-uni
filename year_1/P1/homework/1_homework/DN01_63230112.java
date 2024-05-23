import java.util.Scanner;

public class DN01_63230112 {

    public static void main (String[] args) {
        Scanner sc = new Scanner(System.in);

        int a = sc.nextInt() - 1;
        int b = sc.nextInt() - 1;

        int minSide = b > a ? a : b;
        int count = 0;

        for (int i = 0; i < minSide; i++, b--, a--) {
            count += b * a;
        }

        System.out.println(count);

        sc.close();
    }
}