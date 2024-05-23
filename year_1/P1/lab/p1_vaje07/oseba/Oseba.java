public class Oseba {

    private String ime;
    private String priimek;
    private char spol;
    private int letoRojstva;
    private Oseba oce;
    private Oseba mati;

    public Oseba(String ime, String priimek, char spol, int letoRojstva, Oseba oce, Oseba mati) {
        this.ime = ime;
        this.priimek = priimek;
        this.spol = spol; // M | Z
        this.letoRojstva = letoRojstva;
        this.oce = oce;
        this.mati = mati;
    }

    public String vrniIme() {
        return this.ime;
    }

    @Override
    public String toString() {
        return String.format("%s %s (%s), %d", this.ime, this.priimek, this.spol, this.letoRojstva);
    }

    public boolean jeZenska() {
        return this.spol == 'Z';
    }

    public boolean jeMoski() {
        return this.spol == 'M';
    }

    public int starost(int leto) {
        return leto - letoRojstva;
    }

    public boolean jeStarejsaOd(Oseba os) {
        return this.letoRojstva < os.letoRojstva;
    }

    public static Oseba starejsa(Oseba a, Oseba b) {
        if (a.letoRojstva == b.letoRojstva) {
            return null;
        }

        return a.jeStarejsaOd(b) ? a : b;
    }

    public String imeOceta() {
        return this.oce == null ? null : this.oce.ime;
    }

    public boolean jeBratAliSestraOd(Oseba os) {
        if (this.oce == null || os.mati == null || this.mati == null || os.mati == null || this == os) {
            return false;
        }

        return this.oce == os.oce && this.mati == os.mati;
    }

    public boolean jeSestraOd(Oseba os) {
        return this.jeZenska() && this.jeBratAliSestraOd(os);
    }

    public boolean jeTetaOd(Oseba os) {
        if (os == this) return false;

        if (os.oce == null || os.mati == null) return false;

        return (this.jeSestraOd(os.oce) || this.jeSestraOd(os.mati)) && this.jeZenska();
    }

    public boolean jeOcetovskiPrednikOd(Oseba os) {
        if (os.oce == null) {
            return false;
        }

        if (this == os.oce) {
            return true;
        }

        return this.jeOcetovskiPrednikOd(os.oce);
    }

    public boolean jePrednikOd(Oseba os) {
        if (os == null) return false;

        if (this == os) return true;

        return this.jePrednikOd(os.oce) || this.jePrednikOd(os.mati);
    }
}
