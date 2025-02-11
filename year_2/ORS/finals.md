# Vprašanja

---

Z evolucijo naprav priklopljivih na PCIe začnejo naloge DMA-ja prevzemati kar
naprave same, torej so gospodarji vodila.

---

Stran je lahko velika kot blok, zato da lažje upravljamo s praznimi prostori v PP

---

Pošlje hold request (deaktivira HOLD)
Prekinitev

---

Pri DRAM Memory controller upravlja s signali RAS, CAS in WS. CPE mu samo pošlje
naslov z nastavljanjem RAM-a se ukvarja memory controller.

---

T_RAS - minimalen čas aktivnosti RAS, da preberemo ali zapišemo v pomnilniško lokacijo
T_RCD - RAS-to-CAS delay -> čas potreben, da se informacija iz vrstice v banki zapiše v
register vrstice ... tipalni ojačevalnik
T_RP - row precharge -> čas potreben, da se informacija iz tipalnega ojačevalnika
zapiše nazaj v vrstico v banki
T_RC - row cycle time -> koliko časa rabimo, da beremo/pišemo (T_RAS + T_RP)
t_CL - čas od takrat ko vklopimo CAS signal do takrat ko so podatki dostopni na vodilu

**Kako pohitrimo preslikovanje navideznih naslovov?**
V MMU vgradimo majhen čisto-asociativni (TLB) pomnilnik. Zaradi načela časovne
in prostorske lokalnosti, bomo do naslednjega PA (physical address) z veliko verjetnostjo
našli v isti tabeli kot prejšnjega. Torej bo enak PD (page directory) in PT (page table)
kot prejšnji. V TLB-ju torej hranimo 8 sosednjih preslikavo PA -> FN (frame number)

**Predpostavite, da imate dva enaka DIMM modula? Kako jih boste vstavili v sockete
na matični plošči? Zakaj?**
Oba bomo vstavili v socket 0 ali socket 1, da imata vsak svoj kanal in si ne delita
istega podatkovnega vodila.

**Kako izboljšamo odzivnost DRAM pomnilnikov? Kaj je Fast Page Mode DRAM? Kaj pa EDO DRAM?**
Odzivnost DRAM pomnilnikom izboljšamo tako, da če dostpamo do več zap. stolpcev
v isti vrstici, ne rabimo izvesti row precharge in zapirati RAS, ampak samo
dostopamo do naslednjih stolpcev.

FPM DRAM -> omogoča hitrejši dostop do podatkov v isti vrstici, če je podatek v
isti vrstici, ne rabimo podatka o naslovu. Ko enkrat odpremo vrstico se informacija
shrani v tipalnih ojačevalnikih, tako lahko izvedemo več dostopov do različnih stolpcev
iste vrstice. Vseeno moramo počakati T_CP, da se zapre stolpec in lahko odpremo novega.

EDO DRAM -> podoben FPM, vendar data output ni onemogočen, ko gre CAS high (ga ugasnemo).
To omogoča delno prekrivanje, saj je prejšnji podatek še vedno aktiven na
output pinih, ko že imamo nov naslov stolpca.

**Zakaj se uporablja tok več podatkov, če imamo več dimm channelov?**
Ker imamo lahko multicore CPU.

**Koliko polj DRAM vsebuje ena banka?**
DRAM čip ima lahko 4-16 DRAM polj, do katerih dostopamo istočasno. Preberemo ali
zapišemo število bitov enako številu polj na banki

**Kaj so izjeme (prekinitve in pasti)**
Dogodki, ki prekinejo trenutno izvajanje procesorja in le ta začne izvajati
prekinitveno servisni podprogram. Po koncu se vrne na prekinjeni program.

Pasti -> pride do internega dogodka v CPE med izvajanjem ukazov (neveljaven ukaz, deljenje z 0...)
Prekinitve -> prožene s strani V/I naprav z aktivacijo IRQ

**Zakaj potrebujemo DMA krmilnike? Kako bi bilo brez njih?**
DMA krmilnik omogoča V/I napravam, ad pišejo/berejo iz RAMa brez, da bi dodatno
obremenjevali CPE. DMA tvori naslove in določa posamezne operacije med prenosom (r/w)
Brez njih bi to nalogo moral opravljati CPE.

**Kako se prožijo prekinitve?**
aktiviramo interrupt request signal (IRQ)

**Kakšne izboljšave prinaša SDRAM?**
Dodan končni avtomat in clock. S tem se uskladi prenos podatkov med procesorjem
in pomnilnikom, tako se zmanjša zamuda in izboljšuje učinkovitost prenosa podatkov.
Cilj je bil, da bi bil SDRAM bolj samostojen (sam generiral CAS, RAS, refresh...),
da mu CPE in V/I naprave podajo le address besede. Svojo uro ima, ker dela počasneje
kot procesor. Zaradi ure ve tisti, ki naslavlja RAM točno, kdaj bo dobil podatke.

**Opišite vlogo LAPIC in IO APIC?**
APIC - Advanced Programmable Interrupt Controller

Vsako jedro ima svoj LocalAPIC, ki zna prebrati IO APIC sporočilo ter ustrazno reagirati.
Na IO APIC so vezani vsi IRQ pini. Vsako prekinitev pošlje na najmanj zasedeno
CPU jedro

**Opišite večnivojsko ostranjevanje?**
Za primer uzamemo dvonivojsko ostranjevanje. V RAMu hranimo eno tabelo (4kB), ki
vsebuje naslove vseh ostalih tabel. Njen naslov hranimo v CR3 registru v MMU.
S pomočjo prvih 10 bitov VA (virtual address) izberemo v kateri tabeli se nahaja
naš VA, naložimo to tabelo v RAM in z 10-19 bitom VA poiščemo mesto v tabeli, kjer
se nahaja PA (physical address).
Zaradi velike prostorske lokalnosti se bo nalaganje drugonivojske tabele izvedlo malokrat.

**Kako je V/I vhodna naprava povezana z DMA?**
Na DMA je povezana preko kanala

**Kako se pri SDRAM-ih mapirajo naslovi iz CPE v naslove vrstice, stolpca, banke?**
Z izkoriščanjem prepletanja bank -> ko zmanjka stolpca, gremo v drugo banko namesto
activate new row
Z izkoriščanjem prepletanja blokov v predompnilniku -> zaporedni bloki so v drugih bankah

**Kako je določena frekvenca ure na vodilu za DDR(2,3,4)?**
Fronte -> zaradi zamika ur nastane več front oz. je frekvenca hitrejša.
Glede na Xn-prefetch.

**Kako DMA krmilnik ve v katero smer bo potekal prenos?**
To mi zapišemo v CR - configuration register

**Opišite kaskadno vezavo prekinitvenih krmilnikov 8259A.**
Master in slave imata obvezno različna offseta PSP-jev.
Kaskadna vezava je dobra rešitev, ker imamo večje število IRQ pinov, vendar
kaskadna vezava ne podpira več CPE-jev in prioritete so fiksno določene.
Omogoča 15 prekinjajočih pinov.

1. SLAVE aktivira INT in zato ga tudi MASTER
2. CPE zaključi izvajanje programa in pošlje INTA
3. Po 4 urinih periodah pošlje INTA še enkrat
4. MASTER je sprogramiran tako, da se zaveda kaskadne vezave, zato samo SLAVE
   na podatkovno vodilo zapiše offset

**Kakšna je razlika med fly-by in fly through DMA krmilnik? Navedite dva realna primera.**
FLY-BY -> podatki se ne shranijo v DMA (naprave same izvedejo prenos) - ATA/SATA diski
FLY-THROUGH -> podatki se naložijo v interne FIFO vrste DMA krmilnika in nato
en za drugim prepišejo v ram - počasnejši - Zvočne kartice

**Do katerih DRAM celic v DRAM banki dostopamo istočasno?**
Do celic v isti vrstici in polju

**Opišite inicializacijo DMA krmilnika. Kaj je vse treba nastaviti pred začetkom prenosa?**
Potrebno je konfigurirati 4 registre:

MAR - memory address register (naslov pomnilnika)
PAR - peripheral address register (naslov periferne naprave)
NDR - number of data register (koliko podatkov naj se prenese)
CR - configuration reigster (pove v katero smer naj gre prenos)

**Kaj je osnova ideja pri APIC?**
APIC igra vlogo organizoatorja prekinitev
Potreben je ker so se pojavili sistemi z več CPE jedri
Poleg tega imamo zdaj programabilno določene prioritete

**Kaj je DIMM modul?**
DIMM modul je povezanih več DRAM čipov skupaj.
Imamo 2 ranka, ki si delita vse ukazne signale razen #CS (Chip select). Na enkrat je
lahko aktiven le en. Povečamo kapaciteto. Ranka lahko prepletamo da mskiramo čas dostopa.

**Kaj so to PCI prekinitve? Kako si PCI naprave delijo prekinitvene signale?**
Prekinitve, ki prihajajo iz PCI naprav (zvočne, internetne, zvočne kartice...)
Več naprav si lahko deli isto prekinitveno linijo, kdo je prekinil se ugotovi s pollingom
PIRQA - gor priključimo pomembne naprave
PIRQD - gor priključimo manj pomembne naprave

**Kaj je polje DRAM? Kako je organizirano?**
DRAM - Dynamic Random Access Memory

- 2D struktura
- imamo stolpce in vrstice
- naenkrat lahko odpremo/zapremo eno vrstico
- dostopamo lahko le do stolpcev odprte vrstice
- v enem polju je 1 - bitna pomniška beseda
- potrebno osveževanje

Do podatkov lahko dostopamo z naslovom (RAS - row address strobe) in CAS (column address strobe)
Podatki se prenesejo v register vrstice, kjer ga lahko preberemo.

**Na koga se nanaša naslov, ki ga izstavi fly-by DMA krmilnik?**
Na RAM, V/I napravo naslovimo tako, da ji aktiviramo DACK signal

**Predpostavite, da v sistem želite vstaviti neko PCI kaertico. Kako boste izbrali
na katerem vhodu (PIRQA-PIRQD) bo prožila prekinitve?**
Odvisno v kateri slot jo bomo postavili in odvisno od same PCI naprave, katere funkcionalnosti podpira, mogoče ne proži vseh 4-ih INT-ov.
Procesor to ugotovi, tako da programsko izprašuje INTA A do D.

**Najmanj koliko tabel strani moramo hraniti v pomnilniku pri
2(3,4, ..)-nivojskem ostranjevanju?**
Na prvem nivoju imamo 1, potem jih imamo na drugem nivoju toliko kolikor je velikost
tabele na prvem nivoju in tako naprej.

**Kako lahko fly by naredi mem-mem prenose?**
Uporabi 2 DMA krmilnika.

**Kakšna je razilka med tokom (stream) in kanalom pri DMA krmilniku STM32F**
Vsak stream ima 8 kanalov. Kanali so pini, ki zahtevajo prenos. Vsak Stream
vsebuje svojo FIFO, katere V/I naprave pišejo podatke. FIFO je dober, ker omogoča
8 aktivnih prenosov, torej ni treba (zaporedno) čakati, da RAM konča z pisanjem.
Posledično imamo lahko hitrejše ure na perifernem vodilu.

**Kako osvećujemo vsebino vrstice v DRAM banki?**
Zaporedno odpremo vse vrstice preko RAS signala in jim ponovno vpišemo vrednost.
Ko se vrstice odprejo se načeloma samo ozvežijo s sense amplifier-jem.

**Če DMA ne ve nič o V/I napravi, kako pa pol ve kam/od kje more pošilat?**
Ustrezno je potrebno nastaviti registre MAR, PAR, NDTR, CR - pove v katero smer pošiljamo.

**Kako si ranka delita podatkovne signale?**
Delita si vse podatkovne signale, saj je zaradi CS aktiven le eden od njiju.

**Kako je pametno povezati INTA-INTD signale med posameznimi PCI napravami?**
Uporabimo round-robin routing

**Kaj so to MSI prekinitve?**
MSI - Message Signalled Interrupt

To so prekinitve, ki jih tvorijo V/I naprave same in jih pošiljajo po PCIe vodilu
do LAPIC-ov. Skrajšamo latenco, ker ni več programskega izpraševanja.

**Kako je definiran čas dostopa do vrstice T_RC?**
T_RC = T_RAS + T_RP (read cycle)
Čas, da se prebere ali zapiše informacija in je pripravljena na ponovno branje/pisanje.

**Kam se vežejo PCI prekinitveni signali PIRQA - PIRQD na IO APIC?**
Na IRQ 16 - IRQ 19

**Kako je organizirana prekinitvena tabela pri ARM Cortex M procesorjih?**
ARM Cortex-M jedra imajo:
- do 16 notranjih izjem
- do 240 zunanjih izjem (preko IRQ pinov)

Vsaka izjema ima ID in prioriteto. Nižja številka pomeni višjo prioriteto. Če sta
enaki se vzame tista z nižjim ID. Ima tabelo prekinitvenih vektorjev, ki se začne na 
naslovu x0000_0000.

**Kako preslikujemo naslove v prisotnosti predpomnilnika?**
Najprej pogledamo v TLB L1 predpomnilnika. Če je zadetek smo končali (dobili željeni TAG),
drugače pa po principu dvonivojskega ostranjevanja.

**Kaj so ukazi pri SDRAM-ih?**
Inicializacija:
- AutoRefresh
- Precharge all
- NOP

Menipulacija podatkov:
- Activate (odpre vrstico)
- Read
- Write
- Precharge (pripravi vse vrstice na branje/pisanje)

**Opišite delovanje krmilnika Intel 8259A**
1. V IRR (interrupt request register) se vpišejo prekinitve
2. IMR (interrupt mask register) nadzira katere prekinitve maskiramo oz. ignoriramo
3. Aktiviramo INTR signal, ki gre do CPE
4. Ko CPE zaključi izvajanje ukazov v cevovodu aktivira INTA signal
5. Interrupt kontroler spusti v ISR (interrupt service rutine) bit iz IRR z najvišjo prioriteto
6. To gre skozi kodirnik v izhodni register in z INTA signalom na podatkovno vodilo

**Kaj vsebuje tabela strani?**
Deskriptorje

**Opišite delovanje IO APIC krmilnika**
Odgovoren je za preusmerjanje prekinitev iz zunanjih naprav do LAPIC, ki ga ima
vsako jedro CPU. LAPIC in IO APIC komunicirata preko 3 žičnega APIC bus-a.
Ko se proži eden od 24ih IRQ bitov, se prebere istoležni zapis v tabeli. Nato se
tvori APIC sporočilo, ki vsebuje vse pomembno, naslov cilnjega LAPIC-a, številko
pina, ki je bil prožen in še nekaj ostalih nastavitev

**Opišite dostop (pisalni ali bralni) do banke v SDRAM pomnilniku**
1. Aktivacija vrstice (RAS) - izbere se vrstica in naloži v medpomnilnik
2. Branje/Pisanje (CAS) - Izbran stolpec se prebere ali prepiše
3. Predpolnitev - zapre se vrstica za ponovni dostop

**Do koliko podatkov naenkrat lahko dostopamo pri DDR(2,3,4) DIMM modulu?**
Ponavadi imamo 8 čipov povezanih v rank in vsak vrne 8 bitov. Torej lahko
naenkrat lahko dostopamo do 64 bitov oz 8 byte-ov (en blok).

**Kaj je banka v DRAM pomnilnikih?**
Banka je več DRAM polj vezanih vzporedno. Navadno toliko, da tvorijo dolžino ene besede.

**Kakšna je razlika med eksplozijskim prenosom in 2n-prefetchom? Ali lahko uporabomo oboje?**
Eksplozijski prenos (burst transfer) - Več zaporednih podatkovnih besed se prenese
                                       v enem dostopu brez ponovnega naslavljanja
2n-prefetch - SDram vnaprej prebere 2, 4, 8 ali več podatkivnih enot na en ukaz za
              hitrejši prenos

Lahko uporabljamo oboje, saj 2n-prefetch omogoča hitrejše zajemanje podatkov, ki se
nato prenašajo z eksplozijskim prenosom.

**Kako so kanali označeni na matičnih ploščah?**
Dve isto obarvani reži na matični plošči predstavljata različna kanala.
Če imamo dva rama ju damo na isto barvo za večji bandwidth.

**Kaj je eksplozijski prenos?**
Način prenosa podatkov, kjer se več zaporednih podatkovnih enot prenese v enem
dostopu, bres ponovnega naslavnljana vsake enote posebej. To poveča hitrost
prenosa v pomnilniku in vodilih.

**Opišite dostop (bralni ali pisalni) do DRAM banke (časovno zaporedje naslovnih in
kontrolnih signalov, časi, ...)**
1. Aktivira se RAS signal, ki nam pove katero vrstico želimo odpreti
2. Po času T_RAS (RAS-to-CAS) se s pomočjo CAS signala izbere še stolpec.
3. Po času T_CL (CAS latency) je vsebina pomnilniških celic v vrstici RAS in
   stolpcu CAS v registru vrstice in jo lahko preberemo.
4. Precharge - zaprtje vrstice za nov dostop

**Kaj je napaka strani?**
Pri MMU je to napaka, ko pride do težave pri pravajanju virtualnih naslovov v fizične

**Kako velika naj bo stran?**
Toliko, kolikor so veliki bloki v RAMu in PP (4,16,64kB)

**Kaj je stran in kaj je okvir?**
Stran -> blok v navideznem naslovnem prostoru
Okvir -> blok v fizičnem naslovnem prostoru

**Opišite delovanje DMA krmilnika v sistemih STM32H7**
Omogoča do 8 hkratnih prenosov (vsak ima svoj FIFO).
Arbiter določa in razvršča prenose glede na prioriteto.
Lahko imamo do 128 naprav. DMA krmilnik je viden kot 8x4 reg: MAR, PAR, NDTR, CR

1. CPE nastavi naslov v MAR
2. CPE nastavi naslov v PAR
3. CPE nastavi št. besed, ki jih moramo prenesti z enim DMA prenosom v NDTR reg.
4. CPE v CR nastavi smer prenosa
5. CPE zažene DMA prenos in pri njem ne sodeluje

**Kako maskiramo CL Čas pri SDRAM-u?**
Z burst transferji in prefetchanjem.

**Zakaj imamo v polju DRAM celic dolge vrstice?**
Ker si želimo čim krajše bitne linije, ker je težko zaznavati majhne napetosti n
dolgi liniji.

**Zakaj ni dobro, da v predpomnilbnik gremo z navideznim naslovom? Zakaj pa bi bilo to dobro?**
Isti navidezni naslov se lahko (v dveh različnih programih) preslika v različna
fizična naslova (hononimi)

Preslikovanje in vsa pripadajoča kolobocija ne bi bila potrebna.

**Kako CPE ve, da nekdo prekinja?**
IRQ vhod ali polling

**Kaj je preusmeritvena tabela v IO APIC?**
To je tabela, ki za vseh 24 IRQ pinov hrani 64bitno sporočilo, ki vsebuje ciljni
CPE, št. prekinitvenega vektorja ter prioriteto.

**Kaj je to prekinitvena tabela?**
Tabela, ki vsebuje vse naslove PSP-jev.

CPE lahko pridobi naslov PSP na dva načina:
1. PC = M\[definiran naslov + 4*ID\]
Uporabljamo tabelo prekinitvenih vektorjev. Ta ima na naslovih shranjene naslove PSP-jev.
2. PC = definiran naslov + 4*ID
Uporabljamo prekinitvene vektorje. To so naslovi, na teh naslovih pa imamo jump ukaze
na naslove PSP-jev

**Kako DMA "fly through" ve, da bo imel prenos**
Dobi IRQ

**Kaj se sproži po koncu DMA prenosa?**
Sproži se prekinitev, da CPE obvestimo o končanem prenosu.

**Kaj so kanali? Koliko kanalov podpirajo sodobni procesorji in njihovi 
pomnilniški krmilniki?**
Kanal je več povezanih rankov, tako da se poveže dva DIMM-a, ki si ne delita signalov,
vendar je lahko samo en aktiven naenkrat. Ostane eno naslovno in kontrolno vodilo in
podvojeno podatkovno vodilo.

V današnjih računalnikih lahko do dva DIMM modula tvorita en kanal. In večina
sodobnih računalnikov podpira dva kanala.

**Kako bi s prekinitvenim krmilnikom 8259A servisirali več kot 8 prekinitvenih zahtev (kanalov)?**
S kaskadno vezavo.

**Zakaj potrebujemo signala CAS# in RAS#? Zakaj preprosto ne izstavimo naslova
pomnilniške besede?**
Ker je DRAM organiziran v 2D polje in zaradi kondenzatorjev, moramo najprej izbrati
pravilno vrstico, da se naboj prenesa na bitno linijo. Štele nato lahko izberemo
vrstico, kjer se informacija prenese na register vrstice. Hkrati nam več stolpcev
omogoča "row hit", kjer ne rabimo spreminjati aktivne vrstice in tako hitreje
dostopamo do podatkov.

**Opišite kako pohitrimo dostope pri DDR(2,3,4) v primerjavi s SDRAM-i?**
V DDR uporabimo obe fronti ter uporabljamo Xn-prefetch
Z SDRAM pohitrimo dostope tako, da dostopamo, do več bank hkrati in uporabljamo
uro, ki natančno sinhronizira delovanje pomnilnika

**Zakaj zadošča tako mali TLB? Na kaj se zanašamo?**
Časovno in prostorsko lokalnost in na to, da vsak zapis predstavlja večji kos podatkov.

**Kaj pomeni PC4-19200 pri DDR4 DIMM modulih?**
Ko je DDR SDRAM zapakiran kot DIMM modul, ga označimo z največjim DIMM bandwidthom.
To številko dobimo kot 1200 MHz * 2(DDR) * 8 = 19200 MB/s

**Kaj je deskriptor strani?**
Deskriptor je en vnos v strani (pagetable). Sestavljen je iz:
P - present bit
V - valid bit
FN - frame number
RWX - read write execute

**Kaj je prefetch? Kaj poskušamo s tem pohitriti? Če se razširi vodilo med registrom
vrstice in izhodnim registrom, kaj se more zgoditi v sistemu na podatkovnem
vodilu da accomodata to spremembo?**
Prefetch je tehnika, kjer v eni periodi ure dobimo več podatkov iz SDRAM naenkrat
S tem poskušamo maskirati CL, saj če že imamo odprto vrstico, jo čim bolj izkoristimo
Povečati se mora frekvenca ure, da lahko potem časovno multipleksiramo podatke v večji register

**Ali pri DDR(2,3,4) lahko opravljamo eksplozijski dostop dolžine 1?**
Ne, saj se naenkrat prenesejo le 2, 4 ali 8 bitov.

**Kaj je DMA kanal?**
Poseben par kontrolnih signalov med periferno napravo in DMA kontrolerjem
DMA request (DREQ) in DMA acknowledge (DACK)

**Kako CPE pridobi naslov PSP?**
- Če uporabljamo tabelo prekinitvenih vektorjev:
PC <- M\[definiran naslov + 4 * ID prekinitve\]

- Če uporabljamo prekinitvene vektorje:
PC <- definiran naslov + 4 * ID

- ARM Cortex-M
PC <- M\[0x00000000 + 4 * ID prekinitve\]

- Intel
PC <- M\[0x00000000 + 4 * ID prekinitve\]

- RISC-V
MTVEC hrani pravilo, kako tvorimo naslov PSP:
"00" - direktni način (PC <- BASE * 00) - uporabljamo switch case
"01" - vektorski način (PC <- BASE + 4 * MCAUSE\[9:0\])

**Kaj je prekinitveno servisni podprogram (interrupt handler)?**
PSP je program, ki ga napiše uporabnik in je odvisen od funkcionalnosti periferne naprave
Preden vstopimo v PSP je potrebno shraniti kontekst (shraniti registre za katere
je moćno da jih bo PSP umazal)
Pri RISC-V je potrebno to narediti ročno

**Kaj je CAS latenca pri SDRAM-ih? Kako je pri SDRAM-ih definirana?**
T_CL = CAS latenci je zamik (št. urinih ciklov) med prejetjem read zahteve in
       dobivanja output podatkov

**Kakšne so tipične vrednosti časov T_RCD, T_CL, T_RP pri modernih SDRAM-ih?
Ali jih lahko tehnološko skrajšamo in kako?**
Tipična verednost: 13ns
Skozi leta se ta vrednost ni veliko zmanjšala, zaradi same fizike kondenzatorjev
Spremenile so se samo tehnike za pohitritev prenosa (burst, prepletanje, EDO)

**Kaj je DDR? Kaj je 2n-prefetch?**
DDR - double data rate
Z enim read ukazom se izvedeta 2 in zapišeta v FIFO register. Polovica bitov
se prenese ob pozitivni urini fronti ob negativni pa še druga polovica. Zaradi
težke zaznave front ure, smo dodali še invertiran urin signal. Imamo še 2 nova
signala DQS (data strobe) in DM (data mask)
DQS - sinhroniziran z uro MC, s pomočjo tega se berejo podatki
DM - kontrolira če se podatki prepišejo ali ne

2n-prefetch je dostop do dveh stolpcev naenkrat, ne moremo delati malih burst-ov,
saj je minimum dostop do 2 stolpcev naenkrat -> s časovnim multipleksiranjem
dobimo 2 bita podatkov

**Kaj je rank?**
Rank je vzporedno povezanih več enakih DRAM čipov, ki si delijo isti CS# signal

**Kaj pomenijo časi podatni kot npr. 9-9-9 pri DIMM modulih?**
Časi za T_CL, T_RCD, T_RP

**Kaj je prekinitveni krmilnik in zakaj ga potrebujemo?**
Naprava, ki posreduje CPE informacije o tem, kdo jo prekinja.

Lahko se zgodi, da imamo več V/I naprav, ki želijo prekiniti CPE. Ker jih je lahko 
zelo veliko, jih je težko vse zvezati na CPE, da bi vsak imel svoj IRQ. Zato upoorabljamo
prekinitveni krmilnik, ki vse zunanje interrupt requeste kombinira v eno IRQ linijo
ter jih prioritizira. Tako polling ni potreben.

**Kaj je APIC vodilo? Čemu je namenjeno?**
Prenosu APIC sporočil od IO APIC do LAPIC-ov, ter LAPIC-LAPIC (forwarding)
Je zelo hitro vodilo, iz 3 bitnih linij (CLK + 2 podatkovni)

**Kaj je to navidezni pomnilnik? Zakaj ga imamo?**
Pomnilniški prostor, ki ga vidi progam/uporabnik. S tem omogočimo prenosljivost
programov na sisteme z različnim HW.

1. Želimo da programi in CPE nič ne vedo o količini fizičnega RAM pomnilnika
2. Vsi programi mislijo, da je celoten pomnilniški prostor neke CPE samo njihov.
   Ta program naslavlja poljubne besede v pomnilniškem prostoru in misli, da 
   so samo njegove
3. da so programi pozicijsko neodvisni

**Kako bi povečal PP če imamo samo 12bitov na voljo?**
Damo več PP z istimi velikostmi blokov in temu rečemo set-asociativni predpomnilnik

**Prekinitveni vektorji vs prekinitvene tabele**
Prekinitvena tabela lahko vsebuje prekinitvene vektorje ali pa prvi ukaz
PSP (jump ukaz)
