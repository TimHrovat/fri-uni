import java.util.*;

public class Primer {

    public static void main(String[] args) {
        List<Cas> zbirka = new ArrayList<>();

        zbirka.add(new Cas(5, 10));
        zbirka.add(new Cas(6, 10));
        zbirka.add(new Cas(7, 10));
        zbirka.add(new Cas(8, 10));
        zbirka.add(new Cas(10, 10));

        System.out.println(zbirka.contains(new Cas(7, 10)));
    }

}
