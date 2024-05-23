import java.util.Scanner;
import java.util.Map;
import java.util.TreeMap;
import java.util.ArrayList;

public class DN05_63230112 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int n = sc.nextInt();
        Field field = new Field();

        for (int i = 0; i < n; i++) {
            int shapeIndex = sc.nextInt();
            int x = sc.nextInt();

            Shape shape = ShapeFactory.getShape(shapeIndex, x);

            field.addShape(shape);
        }

        field.print();

        sc.close();
    }
}

class Shape {
    private int x;
    private boolean[][] matrix;

    public Shape(boolean[][] matrix, int x) {
        this.matrix = matrix;
        this.x = x;
    }

    public int getX() {
        return this.x;
    }

    public boolean[][] getMatrix() {
        return this.matrix;
    }

    public int getLowestPointX() {
        boolean[] lastRow = this.matrix[this.matrix.length - 1];

        for (int i = 0; i < lastRow.length; i++) {
            if (lastRow[i]) {
                return x + i;
            }
        }

        return x;
    }

    public ArrayList<Integer> getNotLowestPointXs() {
        int lowestPointx = this.getLowestPointX();
        ArrayList<Integer> notLowestPointXs = new ArrayList<Integer>();

        for (int i = 0; i < matrix[0].length; i++) {
            if (this.x + i != lowestPointx) {
                notLowestPointXs.add(this.x + i);
            }
        }

        return notLowestPointXs;
    }

    public int getEmptyCellsFromBottomByX(int x) {
        int h = 0;

        for (int i = this.matrix.length - 1; i >= 0; i--) {
            if (this.matrix[i][x - this.x]) {
                break;
            }

            h++;
        }

        return h;
    }

    public int getColumnHeightByX(int x) {
        int h = this.matrix.length;

        for (int i = 0; i < this.matrix.length - 1; i++) {
            if (this.matrix[i][x - this.x]) {
                break;
            }

            h--;
        }

        return h;
    }
}

class Field {
    Map<Integer, Long> cols = new TreeMap<Integer, Long>();

    public void addShape(Shape shape) {
        this.addRequiredColumns(shape);

        int lowestPointX = shape.getLowestPointX();
        long lowestPointHeight = cols.get(lowestPointX);
        ArrayList<Integer> notLowestPointXs = shape.getNotLowestPointXs();
        boolean shapeFits = notLowestPointXs.size() == 0 ? true : false;

        while (!shapeFits) {
            for (int x : notLowestPointXs) {
                if (cols.get(x) <= lowestPointHeight + shape.getEmptyCellsFromBottomByX(x)) {
                    shapeFits = true;
                } else {
                    shapeFits = false;
                    lowestPointHeight++;
                    break;
                }
            }
        }

        boolean[][] shapeMatrix = shape.getMatrix();
        int x = shape.getX();

        for (int i = 0; i < shapeMatrix[0].length; i++) {
            long height = lowestPointHeight + shape.getColumnHeightByX(i + x);

            cols.put(x + i, height);
        }
    }

    private void addRequiredColumns(Shape shape) {
        for (int i = 0; i < shape.getMatrix()[0].length; i++) {
            cols.computeIfAbsent(shape.getX() + i, k -> 0L);
        }
    }

    public void print() {
        for (Map.Entry<Integer, Long> column : cols.entrySet()) {
            System.out.printf("%d: %d%n", column.getKey(), column.getValue());
        }
    }

}

class ShapeFactory {
    public static Shape getShape(int shapeIndex, int x) {
        boolean[][] shapeMatrix = getShapeMatrix(shapeIndex);

        return new Shape(shapeMatrix, x);
    }

    private static boolean[][] getShapeMatrix(int shapeIndex) {
        boolean o = false;
        boolean x = true;

        switch (shapeIndex) {
            case 0:
                return new boolean[][] {
                        { x, x, x, x },
                };
            case 1:
                return new boolean[][] {
                        { x },
                        { x },
                        { x },
                        { x },
                };
            case 2:
                return new boolean[][] {
                        { x, x },
                        { x, x },
                };
            case 3:
                return new boolean[][] {
                        { o, x, o },
                        { x, x, x },
                };
            case 4:
                return new boolean[][] {
                        { o, x },
                        { x, x },
                        { o, x },
                };
            case 5:
                return new boolean[][] {
                        { x, x, x },
                        { o, x, o },
                };
            case 6:
                return new boolean[][] {
                        { x, o },
                        { x, x },
                        { x, o },
                };
            case 7:
                return new boolean[][] {
                        { x, o },
                        { x, o },
                        { x, x },
                };
            case 8:
                return new boolean[][] {
                        { o, o, x },
                        { x, x, x },
                };
            case 9:
                return new boolean[][] {
                        { x, x },
                        { o, x },
                        { o, x },
                };
            case 10:
                return new boolean[][] {
                        { x, x, x },
                        { x, o, o },
                };
            case 11:
                return new boolean[][] {
                        { o, x },
                        { o, x },
                        { x, x },
                };
            case 12:
                return new boolean[][] {
                        { x, x, x },
                        { o, o, x },
                };
            case 13:
                return new boolean[][] {
                        { x, x },
                        { x, o },
                        { x, o },
                };
            case 14:
                return new boolean[][] {
                        { x, o, o },
                        { x, x, x },
                };
            case 15:
                return new boolean[][] {
                        { x, x, o },
                        { o, x, x },
                };
            case 16:
                return new boolean[][] {
                        { o, x },
                        { x, x },
                        { x, o },
                };
            case 17:
                return new boolean[][] {
                        { o, x, x },
                        { x, x, o },
                };
            case 18:
                return new boolean[][] {
                        { x, o },
                        { x, x },
                        { o, x },
                };
        }

        return new boolean[0][0];
    }
}