package lecture_2;

import java.util.Scanner;

public class Test {
    static int max_sum = 42;
    static int sum = 0;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        while (Test.sum < Test.max_sum) {
            int new_int;

            try {
                System.out.print("Number input: ");
                new_int = sc.nextInt();
            } catch (Exception e) {
                System.out.println("You must input an integer");
                sc.next();
                continue;
            }

            Test.sum += new_int;
        }

        System.out.println("Sum is " + Test.sum);
        sc.close();
    }
}