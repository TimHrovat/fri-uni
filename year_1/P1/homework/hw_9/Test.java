public class Test {
    public static void main(String[] args) {
        Zival zival = new Zival();
        Sesalec sesalec = new Sesalec();
        Macka macka = new Macka();
        Tiger tiger = new Tiger();
        Lev lev = new Lev();
        Medved medved = new Medved();
        Plazilec plazilec = new Plazilec();
        Zelva zelva = new Zelva();
        Kaca kaca = new Kaca();
        tiger.seHrani();
        medved.seHrani();
        zelva.seHrani();
        plazilec.seHrani();
        System.out.println(zelva.steviloHranjenj()); // 1
        System.out.println(kaca.steviloHranjenj()); // 0
        System.out.println(plazilec.steviloHranjenj()); // 2
        System.out.println(macka.steviloHranjenj()); // 1
        System.out.println(sesalec.steviloHranjenj()); // 2
        System.out.println(zival.steviloHranjenj()); // 4
        System.out.println("---------");

        tiger.preganja(medved);
        sesalec.preganja(kaca);
        zelva.preganja(zival);
        tiger.preganja(medved);
        lev.preganja(lev);
        plazilec.preganja(lev);
        System.out.println(tiger.steviloPreganjanj(medved)); // 2
        System.out.println(macka.steviloPreganjanj(lev)); // 1
        System.out.println(macka.steviloPreganjanj(sesalec)); // 3
        System.out.println(macka.steviloPreganjanj(plazilec)); // 0
        System.out.println(zelva.steviloPreganjanj(sesalec)); // 0
        System.out.println(sesalec.steviloPreganjanj(plazilec)); // 1
        System.out.println(plazilec.steviloPreganjanj(zival)); // 2
        System.out.println(zival.steviloPreganjanj(plazilec)); // 1
        System.out.println(zival.steviloPreganjanj(zival)); // 6
    }
}
