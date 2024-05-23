public class SlikovnaDatoteka extends Datoteka {
    private int sirina;
    private int visina;

    public SlikovnaDatoteka(String ime, int sirina, int visina) {
        this.ime = ime;
        this.sirina = sirina;
        this.visina = visina;
    }

    public int velikost() {
        return 3 * (this.sirina * this.visina) + 54;
    }

    public String toString() {
        return String.format("%s [s %d x %d]", this.ime, this.sirina, this.visina);
    }

    public boolean jeVecjaOd(int prag) {
        return this.visina >= prag && this.sirina >= prag;
    }
}