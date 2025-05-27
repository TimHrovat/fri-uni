package lecture_1;

class Fibonacci {

    public static int valueOf(int index) {
        if (index == 0 || index == 1) {
            return index;
        }

        return valueOf(index - 1) + valueOf(index - 2);
    }
}
