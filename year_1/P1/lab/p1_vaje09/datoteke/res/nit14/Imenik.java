public class Imenik extends Datoteka {
    private Datoteka[] datoteke;

    public Imenik(String ime, Datoteka[] datoteke) {
        this.ime = ime;
        this.datoteke = datoteke;
    }

    public int velikost() {
        int velikost = 256;

        for (Datoteka datoteka : this.datoteke) {
            velikost += datoteka.velikost();
        }

        return velikost;
    }

    public String toString() {
        return String.format("%s [i %d]", this.ime, this.datoteke.length);
    }

    public int steviloVecjihSlik(int prag) {
        int count = 0;

        for (Datoteka datoteka : this.datoteke) {
            if (datoteka instanceof SlikovnaDatoteka && ((SlikovnaDatoteka) datoteka).jeVecjaOd(prag)) {
                count++;
            }
        }

        return count;
    }

    public String poisci(String ime) {
        return this.poisci(".", ime);
    }

    private String poisci(String pot, String ime) {
        for (Datoteka datoteka : this.datoteke) {
            String path = String.format("%s/%s", pot, datoteka.ime);

            if (datoteka.getIme() == ime) {
                return path;
            }

            if (datoteka instanceof Imenik) {
                String found = ((Imenik) datoteka).poisci(path, ime);

                if (found != null) {
                    return found;
                }
            }
        }

        return null;
    }

    public void debugIzpis() {
        this.print(0);
    }

    private void print(int odmik) {
        String padding = "%" + odmik + "s%s%n";
        System.out.printf(padding, "", this.ime);

        for (int i = 0; i < this.datoteke.length; i++) {
            Datoteka datoteka = this.datoteke[i];

            if (datoteka instanceof Imenik) {
                ((Imenik) datoteka).print(odmik + 5);
                continue;
            }

            char character = (i + 1) == this.datoteke.length ? '\\' : '|';
            String formattedString = String.format("%c-- %s", character, datoteka.toString());

            System.out.printf(padding, "",formattedString);
        }
    }
}