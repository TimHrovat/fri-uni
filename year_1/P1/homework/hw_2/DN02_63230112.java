import java.util.Scanner;
import java.lang.Math;
import java.util.HashMap;
import java.util.Map;

// param 1 (b) - oblika tipkovnice (1: ravnovrstnica; 2: kvadratnica; 3: piramidnica; 4: spiralnica)
// param 2 (d) - dimenzija (za ravnovrstnico je d dolžina vrste)
// param 3 (n) - dolžina zaporedja tipk v param 4
// param 4 - zaporedje dolžine n

public class DN02_63230112 {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int keyboardType = sc.nextInt();
        int dimensions = sc.nextInt();
        int sequenceLength = sc.nextInt();
        int sequence[] = new int[sequenceLength];

        for (int i = 0; i < sequenceLength; i++) {
            sequence[i] = sc.nextInt();
        }

        Keyboard keyboard = KeyboardFactory.get(keyboardType);

        int moves = keyboard.countMoves(dimensions, sequence);

        System.out.println(moves);

        sc.close();
    }
}

class KeyboardFactory {
    public static Keyboard get(int keyboardType) {
        switch (keyboardType) {
            case 1:
                return new KeyboardLine();
            case 2:
                return new KeyboardSquare();
            case 3:
                return new KeyboardPyramid();
            case 4:
                return new KeyboardSpiral();
        }

        throw new IllegalArgumentException();
    }
}

interface Keyboard {
    int countMoves(int dimensions, int[] sequence);

    default int calcRectMove(int x1, int x2, int y1, int y2) {
        return Math.abs(x2 - x1) + Math.abs(y2 - y1);
    }
}

class KeyboardLine implements Keyboard {
    @Override
    public int countMoves(int dimensions, int[] sequence) {
        int sum = 0;

        for (int i = 0; i < sequence.length - 1; i++) {
            sum += Math.abs(sequence[i] - sequence[i + 1]);
        }

        return sum;
    }
}

class KeyboardSquare implements Keyboard {
    @Override
    public int countMoves(int dimensions, int[] sequence) {
        int sum = 0;

        for (int i = 0; i < sequence.length - 1; i++) {
            int x1 = sequence[i] % dimensions;
            int x2 = sequence[i + 1] % dimensions;
            int y1 = sequence[i] / dimensions;
            int y2 = sequence[i + 1] / dimensions;

            sum += this.calcRectMove(x1, x2, y1, y2);
        }

        return sum;
    }
}

class KeyboardPyramid implements Keyboard {
    @Override
    public int countMoves(int dimensions, int[] sequence) {
        int sum = 0;
        Map<Integer, Integer[]> coordinates = this.getAllCoordinates(dimensions, sequence);

        for (int i = 0; i < sequence.length - 1; i++) {
            Integer[] key1 = coordinates.get(sequence[i]);
            Integer[] key2 = coordinates.get(sequence[i + 1]);

            sum += this.calcRectMove(key1[0], key2[0], key1[1], key2[1]);
        }

        return sum;
    }

    private Map<Integer, Integer[]> getAllCoordinates(int dimensions, int[] sequence) {
        int maxWidth = (dimensions - 1) * 2 + 1;
        int count = 0;
        Map<Integer, Integer[]> coordinates = new HashMap<Integer, Integer[]>();

        for (int i = 0; i < dimensions; i++) {
            for (int j = maxWidth / 2 - i; j <= maxWidth / 2 + i; j++) {
                coordinates.put(count, new Integer[] { j, i });

                count++;
            }
        }

        return coordinates;
    }
}

class KeyboardSpiral implements Keyboard {
    @Override
    public int countMoves(int dimensions, int[] sequence) {
        int sum = 0;
        Map<Integer, Integer[]> coordinates = this.getAllCoordinates(dimensions, sequence);

        for (int i = 0; i < sequence.length - 1; i++) {
            Integer[] key1 = coordinates.get(sequence[i]);
            Integer[] key2 = coordinates.get(sequence[i + 1]);

            sum += this.calcRectMove(key1[0], key2[0], key1[1], key2[1]);
        }

        return sum;
    }

    private Map<Integer, Integer[]> getAllCoordinates(int dimensions, int[] sequence) {
        int count = 1;
        Map<Integer, Integer[]> coordinates = new HashMap<Integer, Integer[]>();
        coordinates.put(0, new Integer[] { dimensions / 2, dimensions / 2 });

        for (int i = 1; i < dimensions / 2 + 1; i++) {
            int loopDimensions = i * 2 + 1;
            int loopLength = loopDimensions * 4 - 4;
            int x = dimensions / 2 - i;
            int y = dimensions / 2 - i;

            for (int j = 0; j < loopLength; j++) {
                coordinates.put(count, new Integer[] { x, y });

                if (j < loopDimensions - 1) {
                    x++;
                } else if (j < loopDimensions * 2 - 2) {
                    y++;
                } else if (j < loopDimensions * 3 - 3) {
                    x--;
                } else if (j < loopDimensions * 4 - 4) {
                    y--;
                }

                count++;
            }
        }

        return coordinates;
    }
}