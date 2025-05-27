import java.util.*;

public class PrimerTreeSet {
    public static void main(String[] args) {
        // predpostavka da se Cas implementira Comperable
        // vrže izjemo
        // Set<Cas> casi  = new TreeSet<Cas>();
        Set<Cas> casi = new TreeSet<Cas>(Cas.poMinuti());

        // sortirajo se po vrsti zaradi Comparable (naravna urejenost)
        casi.add(new Cas(10, 20));
        casi.add(new Cas(11, 20));
        casi.add(new Cas(9, 20));
        casi.add(new Cas(8, 11));
        casi.add(new Cas(10, 15));
        casi.add(new Cas(10, 26));

        System.out.println(casi);

        System.out.println();

}
