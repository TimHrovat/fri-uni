
public abstract class Zaporedje {

    public abstract Integer y(int x);

    public String vNiz(Interval interval) {
        StringBuilder sb = new StringBuilder("[");
        int zacetek = interval.vrniZacetek();
        int konec = interval.vrniKonec();
        boolean prvic = true;
        for (int x = zacetek;  x <= konec;  x++) {
            Integer y = this.y(x);
            if (y != null) {
                if (!prvic) {
                    sb.append(", ");
                }
                prvic = false;
                sb.append(String.format("%d -> %d", x, y));
            }
        }
        sb.append("]");
        return sb.toString();
    }

    public Interval minMax(Interval interval) {
        int zac = interval.vrniZacetek();
        int kon = interval.vrniKonec();

        Integer min = null;
        Integer max = null;

        for (int i = zac; i <= kon; i++) {
           Integer y = this.y(i);
            if (y == null) {
                continue;
            }

            if (max == null || y > max) {
                max = y;
            } 

            if (min == null || y < min) {
                min = y;
            }
        }
        
        return new Interval(min, max);
    }

    private boolean jeMonotono(Interval interval, int smer) {
        Integer lastY = null;

        for (int i = interval.vrniZacetek(); i <= interval.vrniKonec(); i++) {
            Integer y = this.y(i);

            if (lastY == null) {
                lastY = y;
                continue;
            }
            
            if (y != null) {
                if (lastY != null && y * smer <= lastY * smer) {
                    return false;
                }

                lastY = y;
            }
        }

        return true;
    }

    public boolean jeMonotono(Interval interval) {
        return jeMonotono(interval, 1) || jeMonotono(interval, -1);
    }

    public Zaporedje vsota(Zaporedje drugo) {
        return new Vsota(this, drugo);
    }

    public Zaporedje inverz(Interval interval) {
        if (!this.jeMonotono(interval)) {
            return null;
        }

        return new Inverz(this, interval);
    }
}
