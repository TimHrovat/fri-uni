import java.util.*;

public class Opravki {
    public static void main(String[] args) {
        // hashmap gleda glede na equals (za key!)
        Map<Cas, String> opravki = new HashMap<>();

        opravki.add(new Cas(11, 20), "Zajtrk");
        opravki.add(new Cas(13, 0), "Oma");

    }
}
