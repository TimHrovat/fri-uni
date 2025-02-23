import java.util.Map;
import java.util.HashMap;

public class Tiger extends Bitje {
    private static int stHranjenj = 0;
    private static Map<Integer, Integer> stPreganjanjMap = new HashMap<>();

    private static Bitje parent = new Macka();

    @Override
    public int id() {
        return 7;
    }

    @Override
    public Bitje parent() {
        return parent;
    }

    @Override
    public void seHrani() {
        stHranjenj++;

        try {
            parent.seHrani();
        } catch (Exception e) {
        }
    }

    @Override
    public void preganja(Bitje bitje) {
        Integer stPreganjanj = stPreganjanjMap.getOrDefault(bitje.id(), 0);

        stPreganjanj++;

        stPreganjanjMap.put(bitje.id(), stPreganjanj);

        this.preganjaParent(bitje.parent(), stPreganjanjMap);

        parent.preganja(bitje);
    }

    @Override
    public int steviloHranjenj() {
        return stHranjenj;
    }

    @Override
    public int steviloPreganjanj(Bitje bitje) {
        if (!stPreganjanjMap.containsKey(bitje.id())) {
            return 0;
        }

        return stPreganjanjMap.get(bitje.id());
    }

}
