import java.util.Scanner;
import java.util.Arrays;

public class Izstevanka {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int st_imen = sc.nextInt();
        int izstevanka = sc.nextInt();

        String[] imena = new String[st_imen];

        imena = "Ana Bojan Cene Denis Eva".split("\\s+");

        while (st_imen != 1) {
            int index = izstevanka % st_imen - 1;

            imena = removeIme(imena, index - 1);

            st_imen--;
        }

        sc.close();
    }

    public static String[] removeIme(String[] imena, int index) {
        String[] result = new String[imena.length - 1];

        for (int i = 0; i < imena.length; i++) {
            if (i == index) {
                continue;
            }

            int in = i > index ? i - 1 : i;

            result[in] = imena[i];
        }

        return result;
    }
}
