import java.util.Scanner;

public class DN00_63230112 {

  public static void main(String[] args) {
    int b, p;

    Scanner sc = new Scanner(System.in);

    b = sc.nextInt();
    p = sc.nextInt();

    System.out.println(b / p);
    System.out.println(b % p);

    sc.close();
  }
}
