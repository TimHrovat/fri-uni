
import java.util.*;

public abstract class Lik implements Comparable<Lik> {

    public abstract int ploscina();

    public abstract int obseg();

    public abstract int instanceConst();

    public String toString() {
        return String.format("%s [%s]", this.vrsta(), this.podatki());
    }

    // Vrne vrsto lika <this> (npr. "pravokotnik").
    public abstract String vrsta();

    // Vrne niz s podatki o liku <this>
    // (npr. "širina = 3.0, višina = 4.0").
    public abstract String podatki();

    public static void izpisi(Vektor<Lik> vektor) {
        int stElementov = vektor.steviloElementov();
        for (int i = 0; i < stElementov; i++) {
            Lik lik = vektor.vrni(i);
            System.out.printf("%s | p = %d | o = %d%n",
                    lik.toString(), lik.ploscina(), lik.obseg());
        }
    }

    public static void izpisi(Lik lik) {
        System.out.printf("%s | p = %d | o = %d%n",
                lik.toString(), lik.ploscina(), lik.obseg());
    }

    @Override
    public int compareTo(Lik lik) {
        return Integer.compare(this.ploscina(), lik.ploscina());
    }

    public static Comparator<Lik> poObsegu() {
        return (a, b) -> {
            return Integer.compare(a.obseg(), b.obseg());
        };
    }

    public static Comparator<Lik> poTipu() {
        return (a, b) -> {
            return Integer.compare(a.instanceConst(), b.instanceConst());
        };
    }

    public static void urediPoTipuInObsegu(Vektor<Lik> vektor) {
        Comparator<Lik> comp = Skupno.kompozitum(Lik.poTipu(), Lik.poObsegu());

        Skupno.uredi(vektor, comp);
    }

    public static Lik minKrog(Vektor<Lik> vektor) {
        Comparator<Lik> comp = Skupno.kompozitum(Lik.poTipu(), Lik.poObsegu().reversed());

        Skupno.uredi(vektor, comp);

        return vektor.vrni(vektor.steviloElementov() - 1);
    }
}
