public class Vsota extends Zaporedje {
    Zaporedje zap1;
    Zaporedje zap2;

    public Vsota (Zaporedje zap1, Zaporedje zap2) {
       this.zap1 = zap1; 
       this.zap2 = zap2;
    }

    public Integer y(int x) {
        Integer y1 = zap1.y(x);
        Integer y2 = zap2.y(x);

        if (y1 == null || y2 == null) {
            return null;
        }

        return y1 + y2;
    }


}
