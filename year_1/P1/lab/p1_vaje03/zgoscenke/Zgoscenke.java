import java.util.Scanner;

public class Zgoscenke {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        int k = sc.nextInt();

        int disk_data_size = 0;
        int count = n;

        while (sc.hasNextInt()) {
            int size = sc.nextInt();

            if (disk_data_size + size > k) {
                count--;
                disk_data_size = 0;
            }

            if (count == 0)
                break;

            disk_data_size += size;

            System.out.printf("%d EP -> zgoscenka %d (%d EP)%n", size, (n + 1) - count, disk_data_size);
        }
    }
}
