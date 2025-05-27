import java.util.Scanner;

public class VrstaZbor {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        int[] vrsta = new int[n];
        int countMatches = 0;

        for (int i = 0; i < n; i++) {
            vrsta[i] = sc.nextInt();
        }

        for (int i = 0; i < vrsta.length; i++) {
            if (i < vrsta.length - 1 && vrsta[i] > vrsta[i + 1]) {
                continue;
            }

            if (i > 0 && vrsta[i - 1] > vrsta[i]) {
                continue;
            }

            System.out.println(i);
            countMatches++;
        }

        if (countMatches == 0) {
            System.out.println("NOBEDEN");
        }

        sc.close();
    }
}