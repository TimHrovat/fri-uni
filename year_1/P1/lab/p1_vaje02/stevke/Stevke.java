import java.util.Scanner;

public class Stevke {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        long input = sc.nextLong();

        leftToRight(input);

        sc.close();
    }

    private static void leftToRight(long input) {
        if (input == 0L) {
            return;
        }

        leftToRight(input / 10L);

        System.out.println(input % 10L);
    }

    private static void rightToLeft(long input) {
        while (input != 0L) {
            System.out.println(input % 10L);

            input = input / 10L;
        }
    }
}
