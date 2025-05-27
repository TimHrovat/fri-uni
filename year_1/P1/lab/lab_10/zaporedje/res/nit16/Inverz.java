public class Inverz extends Zaporedje {
    
    Zaporedje zap;
    Interval interval;

    public Inverz(Zaporedje zap, Interval interval) {
        this.zap = zap; 
        this.interval = interval;
    }

    @Override
    public Integer y(int x) {
        int zac = interval.vrniZacetek();
        int kon = interval.vrniKonec();

        for (int i = zac; i <= kon; i++) {
            Integer y = zap.y(i);
            
            if (y != null && y == x) {
                return i;
            }
        }

        return null;
    }
}
