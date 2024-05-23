import java.util.Scanner;
import java.lang.Math;

public class Astroid {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        float r = sc.nextFloat();

        for (float y = r; y >= -r; y--) {
            for (float x = -r; x <= r; x++) {
                if ((Math.pow(Math.abs(x), 2.0/3) + Math.pow(Math.abs(y), 2.0/3)) <= Math.pow(Math.abs(r), 2.0/3)) {
                    System.out.print("*");
                } else {
                    System.out.print(" ");
                }
            }
            System.out.printf("%n");
        }
    }
}
