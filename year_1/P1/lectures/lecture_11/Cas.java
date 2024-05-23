import java.util.*;

public class Cas implements Comparable<Cas> {
    private int ura;
    private int minuta;

    public Cas(int ura, int minuta) {
        this.ura = ura;
        this.minuta = minuta;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }

        if (!(o instanceof Cas)) {
            return false;
        }

        Cas oCas = (Cas) o;

        return this.ura == oCas.ura && this.minuta == oCas.minuta;
    }

    @Override
    public int compareTo(Cas o) {
        int thisMin = 60 * this.ura + this.minuta;
        int drugiMin = 60 * o.ura + o.minuta;

        return thisMin - drugiMin;
    }

    @Override
    public int hashCode() {
        return 7 * Integer.hashCode(this.minuta) + 11 * Integer.hashCode(this.ura);
    }

    @Override
    public String toString() {
        return String.format("%d:%02d", this.ura, this.minuta);
    }

    public static Comparator<Cas> poMinuti() {
        // return new PrimerjalnikPoMinuti();

        return ((a, b) -> {
            return a.minuta - b.minuta;
        });
    }

    private static class PrimerjalnikPoMinuti implements Comparator<Cas> {
        @Override
        public int compare(Cas cas1, Cas cas2) {
            return cas1.minuta - cas2.minuta;
        }
    }
}
