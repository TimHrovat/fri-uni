import java.util.*;

public class PrimerSet {
    public static void main(String[] args) {
        Set<Cas> casi = new HashSet<>();

        casi.add(new Cas(7, 50));
        casi.add(new Cas(10, 50));
        casi.add(new Cas(7, 50));
        casi.add(new Cas(7, 55));

        System.out.println(casi.size());
        System.out.println(casi);
    }
}
