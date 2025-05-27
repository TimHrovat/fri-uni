import java.util.Scanner;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;

public class DN04_63230112 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int stGostov = sc.nextInt();
        int luckyNumber = sc.nextInt();
        Long sumOfPairs = 0L;

        ArrayList<Integer> guestMTags = new ArrayList<Integer>();
        Map<Integer, Long> guestWTags = new HashMap<Integer, Long>();

        for (int i = 0; i < stGostov; i++) {
            int tag = sc.nextInt();

            guestMTags.add(tag);

            Long wTagCount = guestWTags.get(tag);

            if (wTagCount == null) {
                wTagCount = 0L;
            }

            wTagCount++;

            guestWTags.put(tag, wTagCount);
        }

        for (int i = 0; i < stGostov; i++) {
            Long matchingTagsCount = guestWTags.get(luckyNumber - guestMTags.get(i));

            if (matchingTagsCount == null) {
                continue;
            }

            sumOfPairs += matchingTagsCount;
        }

        System.out.println(sumOfPairs);

        sc.close();
    }
}