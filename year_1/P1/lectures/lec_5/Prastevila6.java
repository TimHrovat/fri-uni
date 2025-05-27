import java.util.Scanner;

public class Prastevila6 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();

        boolean[] sestavljeno = new boolean[n + 1];

        int p = 2;
        int koren = (int) Math.round(Math.sqrt(n));

        while (p <= koren) {
            for (int i = 2 * p; i <= n; i += p) {
                sestavljeno[i] = true;
            }

            do {
                p++;
            } while (p <= koren && sestavljeno[p]);
        }

        for (int i = 2; i <= n; i++) {
            if (!sestavljeno[i]) {
                System.out.println(i);
            }
        }
    }
}