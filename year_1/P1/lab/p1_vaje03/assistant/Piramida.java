import java.util.*;

/**
 * n - visina piramide
 *
 *    * 
 *   ***
 *  *****
 *
 */
 

public class Piramida {
	
	private static void znaki(int n, char znak) {
		// for (int i = 0; i < n; i++)
		//	System.out.print(znak);
		// python "*" * 7
		System.out.print("*".repeat(7));
	}
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int n = sc.nextInt();
		
		// v.1
		// znaki(8, '#');
		for (int i = 0; i < n; i++) {
			// izris presledkov
			for (int j = 0; j < n - i - 1; j++)
				System.out.print(' ');
			// izris zvezdic
			for (int j = 0; j < 2 * i + 1; j++)
				System.out.print('*');
			// skok v NV
			System.out.println();
		}
		
		// v.2
		int stPresledkov = n - 1;
		int stZvezdic = 1;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < stPresledkov; j++)
				System.out.print(' ');
			for (int j = 0; j < stZvezdic; j++)
				System.out.print('*');
			System.out.println();
			
			stPresledkov--;
			stZvezdic += 2;
		}
		
		// v.3
		
	}
}