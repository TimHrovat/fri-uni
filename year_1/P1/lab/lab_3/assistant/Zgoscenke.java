import java.util.Scanner;

public class Zgoscenke {
	
	// n - stevilo zgoscenk
	// capacity - velikost CD-ja v EP
	// fileSize - velikost datoteke v EP
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		int n = sc.nextInt(); 
		int capacity = sc.nextInt(); 
		
		int stCD = 1;
		int freeSpace = capacity;
		
		while (sc.hasNextInt()) {
			int fileSize = sc.nextInt();
			if (freeSpace >= fileSize) {
				freeSpace -= fileSize;
			} else {
				
			}
			
			
			// System.out.println(fileSize);
			System.out.printf("500 EP -> zgoscenka 1 (500 EP)%n", ...);
		}
		
		
	}
}