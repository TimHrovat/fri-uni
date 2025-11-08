# Questions

## Exercise 1

### Manual convolution calculation

f = \[0,1,1,1,0,0.7,0.5,0.2,0,0,1,0\]
k = \[0.5,1,0.3\]
k_flip = \[0.3,1,0.5\]

i ... 0-11

i=0: k_flip[0]*f[0] + k_flip[1]*f[-1] + k_flip[2]*f[-2] = 0.3*0 + 1*0 + 0.5*0 = 0
i=1: k_flip[0]*f[1] + k_flip[1]*f[0] + k_flip[2]*f[-1] = 0.3*1 + 1*0 + 0.5*0 = 0.3
i=2: k_flip[0]*f[2] + k_flip[1]*f[1] + k_flip[2]*f[0] = 0.3*1 + 1*1 + 0.5*0 = 1.3
i=3: k_flip[0]*f[3] + k_flip[1]*f[2] + k_flip[2]*f[1] = 0.3*1 + 1*1 + 0.5*1 = 1.8
i=4: k_flip[0]*f[4] + k_flip[1]*f[3] + k_flip[2]*f[2] = 0.3*0 + 1*1 + 0.5*1 = 1.5
i=5: k_flip[0]*f[5] + k_flip[1]*f[4] + k_flip[2]*f[3] = 0.3*0.7 + 1*0 + 0.5*1 = 0.21 + 0 + 0.5 = 0.71
i=6: k_flip[0]*f[6] + k_flip[1]*f[5] + k_flip[2]*f[4] = 0.3*0.5 + 1*0.7 + 0.5*0 = 0.15 + 0.7 + 0 = 0.85
i=7: k_flip[0]*f[7] + k_flip[1]*f[6] + k_flip[2]*f[5] = 0.3*0.2 + 1*0.5 + 0.5*0.7 = 0.06 + 0.5 + 0.35 = 0.91
i=8: k_flip[0]*f[8] + k_flip[1]*f[7] + k_flip[2]*f[6] = 0.3*0 + 1*0.2 + 0.5*0.5 = 0 + 0.2 + 0.25 = 0.45
i=9: k_flip[0]*f[9] + k_flip[1]*f[8] + k_flip[2]*f[7] = 0.3*0 + 1*0 + 0.5*0.2 = 0 + 0 + 0.1 = 0.1
i=10: k_flip[0]*f[10] + k_flip[1]*f[9] + k_flip[2]*f[8] = 0.3*1 + 1*0 + 0.5*0 = 0.3 + 0 + 0 = 0.3
i=11: k_flip[0]*f[11] + k_flip[1]*f[10] + k_flip[2]*f[9] = 0.3*0 + 1*1 + 0.5*0 = 0 + 1 + 0 = 1

[0, 0.3, 1.3, 1.8, 1.5, 0.71, 0.85, 0.91, 0.45, 0.1, 0.3, 1]
[0.3, 1.3, 1.8, 1.5, 0.71, 0.85, 0.91, 0.45, 0.1, 0.3]

### Can you recognize the shape of the kernel? What is the sum of the elements in the kernel? How does the kernel affect the signal?

- gausova krivulja
- 1
- zgladi ostre tranzicije v signalu (smoothing)

### The figure below shows two kernels (a) and (b) as well as signal (c).

Sketch (do not focus on exact proportions of your drawing, but rather on the understanding
of what you are doing) the resulting convolved signal of the given input
signal and each kernel.

TODO

### Convolution associativity effect

Z konvolucijo lahko izboljšamo performance saj lahko več operacij na slikah (npr. edge detection in blur), damo v eno konvolucijo

## Exercise 2

### Question: Which noise is better removed using the Gaussian filter?

gaussian noise

### Which filter performs better at this specific task? In comparison to Gaussian filter that can be applied multiple times in any order, does the order matter in case of median filter? What is the name of filters like this?

Za salt-pepper se bolje obnese medianski filter, ker nadomesti izstopajoče vrednosti z mediano in tako učinkovito odstrani impulzni šum, hkrati pa ohrani zgradbo signala.

Da, median filter je nelinearen, zato je vrstni red pomemben. Nelinearni filtri.

### Which image (object_02_1.png or object_03_1.png) is more similar to image object_01_1.png considering the L2 distance? How about the other three distances? We can see that all three histograms contain a strongly expressed component (one bin has a much higher value than the others). Which color does this bin represent?

Object 01_1 vs Object 02_1:
L2: 0.4263
chi2: 0.4322
intersection: 0.6007
hellinger: 0.5745

Object 01_1 vs Object 03_1:
L2: 0.0951
chi2: 0.1307
intersection: 0.1970
hellinger: 0.3207

črna (ozadje je črno)

### Which distance is in your opinion best suited for image retrieval? How does the retrieved sequence change if you use a different number of bins? Is the execution time affected by the number of bins?

**Najboljša razdalja**: Hellinger razdalja - dobro razlikuje med podobnimi in nepodobnimi slikami, manj občutljiva na osamelce.

**Vpliv števila binov**: Manj binov = manj diskriminativno, vendar bolj robustno. Več binov = bolj diskriminativno, vendar občutljivejše na šum.

**Čas izvajanja**: Da, čas se poveča kubično s številom binov (O(n_bins³)).