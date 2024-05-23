import java.util.*;

public class Test {

    public static void main(String[] args) {
        List<String> nizi = new ArrayList<String>(List.of("hello", "world"));

        metoda(nizi);
    }

    public static <T extends Iterable<String>> void metoda(T nizi) {
        for(String niz : nizi) {
            
        }
    }
}
