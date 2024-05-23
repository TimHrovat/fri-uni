package lecture_1;

import java.util.Scanner;

public class Test {

    public static void main(String args[]) {
        Scanner scanner = new Scanner(System.in);
        int index;

        System.out.print("Vpisi index: ");

        try {
            // if the value is not an integer it throws an exception
            index = scanner.nextInt();
        } catch (Exception e) {
            System.out.println("ERROR");

            scanner.close();
            return;
        }

        System.out.println(Fibonacci.valueOf(index));

        scanner.close();
    }
}
