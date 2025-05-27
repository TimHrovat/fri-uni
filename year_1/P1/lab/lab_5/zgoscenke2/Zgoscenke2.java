import java.util.Scanner;
import java.util.Arrays;

public class Zgoscenke2 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        int k = sc.nextInt();
        int[] zgoscenke = new int[n];

        while (sc.hasNextInt()) {
            int size = sc.nextInt();

            int index = findIndex(zgoscenke, size, k);

            if (index == -1) {
                break;
            }

            zgoscenke[index] += size;

            System.out.printf("%d EP -> zgoscenka %d %s%n", size, index + 1, Arrays.toString(zgoscenke));
        }

        sc.close();
    }

    public static int findIndex(int[] zgoscenke, int zapis, int max_zgoscenka) {
        int max_size = max_zgoscenka - zgoscenke[0];
        int max_size_index = 0;

        for (int i = 1; i < zgoscenke.length; i++) {
            int remaining = max_zgoscenka - zgoscenke[i];

            if (remaining > max_size) {
                max_size = remaining;
                max_size_index = i;
            }
        }

        return max_size < zapis ? -1 : max_size_index;
    }
}
