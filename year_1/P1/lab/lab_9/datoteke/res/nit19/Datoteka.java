public abstract class Datoteka {
    protected String ime;

    public String getIme() {
        return this.ime;
    }

    abstract public int velikost();

    abstract public String toString();
}
