import java.util.Scanner;
import java.util.Random;

public class Berac {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int seed = sc.nextInt();
        int mEURForGBP = sc.nextInt();
        int mEURForUSD = sc.nextInt();
        int maxGift = sc.nextInt();
        int endDayAt = sc.nextInt();

        Random random = new Random(seed);

        int giftsValue = 0;

        System.out.println("   Darilo          Klobuk");

        while (giftsValue < endDayAt) {
            int currency = random.nextInt(3);
            int gift = random.nextInt(maxGift);
            String currencyName = "EUR";

            switch (currency) {
                case 0: // EUR
                    giftsValue += gift * 1000;
                    break;
                case 1: // GBP
                    giftsValue += gift * mEURForGBP;
                    currencyName = "GBP";
                    break;
                case 2: // USD
                    giftsValue += gift * mEURForUSD;
                    currencyName = "USD";
                    break;
            }

            System.out.printf("%5d %s%11d mEUR%n", gift, currencyName, giftsValue);
        }
    }
}
