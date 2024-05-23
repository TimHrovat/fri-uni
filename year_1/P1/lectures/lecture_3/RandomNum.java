package lecture_3;

import java.util.Random;
import java.util.Scanner;

public class RandomNum {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Random rand = new Random();

        int upperLimit = sc.nextInt();

        int randomNumber = rand.nextInt(upperLimit);
        int input;

        do {
            input = sc.nextInt();

            if (input < randomNumber) {
                System.out.println("vecje");
            } else if (input > randomNumber) {
                System.out.println("manjse");
            }
        } while (randomNumber != input);

        System.out.println("pravilno");

        sc.close();
    }
}
