import java.util.Scanner;

public class Piramida {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int piramidaWidth = sc.nextInt();

        for (int i = 0; i < piramidaWidth; i++) {
            String line = "";

            for (int j = 0; j < piramidaWidth - i - 1; j++) {
                line += " ";
            }

            for (int j = 0; j < (i * 2) + 1; j++) {
                line += "*";
            }

            System.out.println(line);
        }

        sc.close();
    }
}