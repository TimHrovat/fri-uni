import java.util.Scanner;

public class Romanje {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int d = sc.nextInt(); // dolzina poti
        int p = sc.nextInt(); // prehodi prvi dan
        int z = sc.nextInt(); // vsak dan toliko enot manj

        int count = 1;

        while (d > 0 && p > 0) {
            System.out.printf("%d: %d -> %d%n", count, d, d - p > 0 ? d - p : 0);

            d -= p;
            p -= z;

            count++;
        }

        sc.close();
    }
}
