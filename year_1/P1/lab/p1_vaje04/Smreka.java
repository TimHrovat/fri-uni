package p1_vaje04;
import java.util.Scanner;

public class Smreka {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int height = sc.nextInt();

        for (int i = 0; i < height; i++) {
            piramida(i + 1, height - (i+1));
        }

        sc.close();
    }

    public static void piramida(int piramidaWidth, int offset) {
        for (int i = 0; i < piramidaWidth; i++) {
            String line = "";

            for (int j = 0; j < piramidaWidth - i - 1; j++) {
                line += " ";
            }

            for (int j = 0; j < (i * 2) + 1; j++) {
                line += "*";
            }

            System.out.println(offsetStr(offset) + line);
        }
    }

    public static String offsetStr(int offset) {
        String res = "";

        for (int i = 0; i < offset; i++) {
            res += " ";
        }

        return res;
    }
}
