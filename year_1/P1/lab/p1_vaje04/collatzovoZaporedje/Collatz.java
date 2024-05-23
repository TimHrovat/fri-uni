import java.util.Scanner;

public class Collatz {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int minLimit = sc.nextInt();
        int maxLimit = sc.nextInt();
        int maxCount = 0;
        int longest = minLimit;

        while (minLimit <= maxLimit) {
            int count = count(minLimit);

            if (count > maxCount) {
                maxCount = count;
                longest = minLimit;
            }

            minLimit++;
        }

        System.out.println(longest);
        System.out.println(maxCount);
    }

    public static int count(int n) {
        int count = 1;

        while (n != 1.0f) {
            n = n % 2 == 0 ? n / 2 : n * 3 + 1;

            count++;
        }

        return count;
    }
}