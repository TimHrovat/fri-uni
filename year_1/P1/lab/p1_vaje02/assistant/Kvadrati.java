import java.util.Scanner;

public class Kvadrati {

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);

		int spMeja = sc.nextInt();
		int zgMeja = sc.nextInt();

		for (int i = spMeja; i <= zgMeja; i++) {
			int kvadrat = i * i;
			System.out.printf("%d%n", kvadrat);
		}
	}
}


