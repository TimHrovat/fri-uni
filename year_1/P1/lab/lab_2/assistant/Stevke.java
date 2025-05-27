import java.util.Scanner;

public class Stevke {
	
	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		
		long number = sc.nextLong();
		
		while (number > 0L) {
			int digit = (int) (number % 10L);
			System.out.println(digit);
			number /= 10L;
		}
		
	}
}