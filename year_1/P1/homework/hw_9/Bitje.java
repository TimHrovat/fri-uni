import java.util.Map;

abstract class Bitje {
    abstract public int id();

    abstract public Bitje parent();

    abstract public void seHrani();

    abstract public void preganja(Bitje bitje);

    abstract public int steviloHranjenj();

    abstract public int steviloPreganjanj(Bitje bitje);

    protected void preganjaParent(Bitje bitje, Map<Integer, Integer> stPreganjanjMap) {
        if (bitje == null) {
            return;
        }

        Integer stPreganjanj = stPreganjanjMap.getOrDefault(bitje.id(), 0);

        stPreganjanj++;

        stPreganjanjMap.put(bitje.id(), stPreganjanj);

        this.preganjaParent(bitje.parent(), stPreganjanjMap);
    }

}
