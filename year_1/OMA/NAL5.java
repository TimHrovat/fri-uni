public class NAL5 {
    public static void main(String[] args) {
        System.out.println(calculateForN(7));
    }

    public static float calculateForN(int clen) {
        if (clen == 1) {
            return 1.0f; // value of a1
        }

        float an_minus_1 = calculateForN(clen - 1);

        return ((an_minus_1 + (12 / an_minus_1))/2);
    }
}