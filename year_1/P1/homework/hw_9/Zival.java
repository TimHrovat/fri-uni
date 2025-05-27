import java.util.Map;
import java.util.HashMap;

public class Zival extends Bitje {
    private static int stHranjenj = 0;
    private static Map<Integer, Integer> stPreganjanjMap = new HashMap<>();

    @Override
    public int id() {
        return 0;
    }

    @Override
    public Bitje parent() {
        return null;
    }

    @Override
    public void seHrani() {
        stHranjenj++;
    }

    @Override
    public void preganja(Bitje bitje) {
        Integer stPreganjanj = stPreganjanjMap.getOrDefault(bitje.id(), 0);

        stPreganjanj++;

        stPreganjanjMap.put(bitje.id(), stPreganjanj);

        this.preganjaParent(bitje.parent(), stPreganjanjMap);
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
