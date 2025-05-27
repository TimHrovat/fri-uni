import java.util.*;
import java.lang.Math;

public class Gravitacija {
    private static final double GRAVITATIONAL_CONSTANT = 6.674e-11d;
    private static final double EARTH_MASS = 5.972e24d;
    private static final double EARTH_RADIUS = 6.371e6d; 
    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        double altitude = sc.nextDouble();
        
        System.out.println(altitude);

        double acceleration = calcGravitationalAcceleration(altitude);

        System.out.println(acceleration);
    }

    public static double calcGravitationalAcceleration(double altitude) {
        return (GRAVITATIONAL_CONSTANT * EARTH_MASS) / 
            Math.pow(EARTH_RADIUS + altitude, 2);

    }
}
