import java.util.Scanner;

public class Anka {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int m = sc.nextInt();
        int d = sc.nextInt();
        int count = 0;

        int[] firstPlacePossibilities = { 1, 3, 5, 7, 9 };

        for (int firstPlace : firstPlacePossibilities) {
            for (int secondPlace = m + 1; secondPlace < 10; secondPlace++) {
                for (int thirdPlace = 0; thirdPlace < 10; thirdPlace += d) {
                    System.out.printf("%d-%d-%d%n", firstPlace, secondPlace, thirdPlace);
                    count++;
                }
            }
        }

        System.out.println(count);
        sc.close();
    }
}