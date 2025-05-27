public class BesedilnaDatoteka extends Datoteka {
    private int stZnakov;

    public BesedilnaDatoteka(String ime, int stZnakov) {
        this.ime = ime;
        this.stZnakov = stZnakov;
    }

    public int velikost() {
        return this.stZnakov;
    }

    public String toString() {
        return String.format("%s [b %d]", this.ime, this.stZnakov);
    }
}