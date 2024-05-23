package lecture_3.prime_numbers;

import java.util.Scanner;

public class PrimeNumbersV1 {

    public static void main(String[] args) {
        Scanner sc = new Scanner (System.in);

        int upperLimit = sc.nextInt();

        for (int i = 2; i < upperLimit; i++) {
            if (PrimeNumbersV1.countDividers(i) == 2) {
                System.out.println(i);
            }
        }

        sc.close();
    }

    private static int countDividers(int num) {
        int count = 2;

        for (int i = 2; i <= num / 2; i++) {

            if (num % i == 0) {
                count++;
            }
        }

        return count;
    }
}
