import java.util.Scanner;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import java.util.Comparator;

public class DN08_63230112 {
    private static Scanner sc = new Scanner(System.in);

    enum PublicationType {
        monografija,
        clanek,
        referat,
    }

    private static class GCD {
        public static int calculate(int a, int b) {
            if (b==0) return a;

            return calculate(b,a%b);
        }
    }

    private static abstract class Publication {
        protected String[] authors;
        protected String title;

        public Publication(String[] authors, String title) {
            this.authors = authors;
            this.title = title;
        }

        public abstract int getPointBase();

        public Float getPoints() {
            return this.getPointBase() / (float) this.authors.length;
        };

        public String getPointsString() {
            int pointBase = this.getPointBase();

            int whole = pointBase / this.authors.length;
            int remainder = pointBase % this.authors.length;

            int gcd = GCD.calculate(remainder, this.authors.length);

            remainder = remainder / gcd;
            int authorCount = this.authors.length / gcd;

            return remainder == 0 ? String.format("%d", whole)
                    : String.format("%d+%d/%d", whole, remainder, authorCount);
        };

        public abstract String toString();
    }

    private static class Monograph extends Publication {
        private String publisher;
        private int releaseYear;
        private String ISBN;

        public Monograph(String[] authors, String title, String publisher, int releaseYear, String ISBN) {
            super(authors, title);

            this.publisher = publisher;
            this.releaseYear = releaseYear;
            this.ISBN = ISBN;
        }

        public int getPointBase() {
            return 10;
        }

        @Override
        public String toString() {
            String authors = String.join(", ", this.authors);
            String points = this.getPointsString();

            return String.format("%s: %s. %s %d, ISBN %s | %s", authors, this.title, this.publisher, this.releaseYear,
                    this.ISBN, points);
        }
    }

    private static class ResearchPaper extends Publication {
        private String conferenceTitle;
        private Boolean conferenceType;
        private int startPage;
        private int endPage;

        public ResearchPaper(String[] authors, String title, String conferenceTitle, Boolean conferenceType,
                int startPage, int endPage) {
            super(authors, title);

            this.conferenceTitle = conferenceTitle;
            this.conferenceType = conferenceType;
            this.startPage = startPage;
            this.endPage = endPage;
        }

        public int getPointBase() {
            return this.conferenceType ? 3 : 1;
        }

        @Override
        public String toString() {
            String authors = String.join(", ", this.authors);
            String points = this.getPointsString();

            return String.format("%s: %s. %s: %d-%d | %s", authors, this.title, this.conferenceTitle, this.startPage,
                    this.endPage, points);
        }
    }

    private static class Article extends Publication {
        private String magazineTitle;
        private int year;
        private int number;
        private int releaseYear;
        private int position;
        private int positionCount;
        private int startPage;
        private int endPage;

        public Article(String[] authors, String title, String magazineTitle, int year, int number, int releaseYear,
                int position, int positionCount, int startPage, int endPage) {
            super(authors, title);

            this.magazineTitle = magazineTitle;
            this.year = year;
            this.number = number;
            this.releaseYear = releaseYear;
            this.position = position;
            this.positionCount = positionCount;
            this.startPage = startPage;
            this.endPage = endPage;
        }

        public int getPointBase() {
            Float positionFactor = this.position / (float) this.positionCount;

            if (positionFactor <= 0.25F) {
                return 10;
            } else if (positionFactor <= 0.5F) {
                return 8;
            } else if (positionFactor <= 0.75F) {
                return 6;
            } else if (positionFactor <= 1.0F) {
                return 4;
            }

            return 2;
        }

        @Override
        public String toString() {
            String authors = String.join(", ", this.authors);
            String points = this.getPointsString();

            return String.format("%s: %s. %s %d(%d): %d-%d (%d) | %s", authors, this.title, this.magazineTitle,
                    this.year, this.number,
                    this.startPage, this.endPage, this.releaseYear, points);
        }
    }

    private static class PublicationFactory {
        public static Publication getPublication(String type, String employee) {
            PublicationType publicationType = PublicationType.valueOf(type);

            switch (publicationType) {
                case PublicationType.monografija:
                    return buildMonograph(employee);
                case PublicationType.referat:
                    return buildResearchPaper(employee);
                case PublicationType.clanek:
                    return buildArticle(employee);
            }

            return null;
        }

        private static String[] getAuthors(String employee) {
            int authorCount = sc.nextInt();
            String[] authors = new String[authorCount];

            for (int i = 0; i < authorCount; i++) {
                String author = sc.next();

                author = author.equals("#") ? employee : author;

                authors[i] = author;
            }

            return authors;
        }

        private static Publication buildMonograph(String employee) {
            String[] authors = getAuthors(employee);
            String title = sc.next();
            String publisher = sc.next();
            int releaseYear = sc.nextInt();
            String ISBN = sc.next();

            return new Monograph(authors, title, publisher, releaseYear, ISBN);
        }

        private static Publication buildResearchPaper(String employee) {
            String[] authors = getAuthors(employee);
            String title = sc.next();
            String conferenceTitle = sc.next();
            Boolean conferenceType = sc.nextBoolean();
            int startPage = sc.nextInt();
            int endPage = sc.nextInt();

            return new ResearchPaper(authors, title, conferenceTitle, conferenceType, startPage, endPage);
        }

        private static Publication buildArticle(String employee) {
            String[] authors = getAuthors(employee);
            String title = sc.next();
            String magazineTitle = sc.next();
            int year = sc.nextInt();
            int number = sc.nextInt();
            int releaseYear = sc.nextInt();
            int position = sc.nextInt();
            int positionCount = sc.nextInt();
            int startPage = sc.nextInt();
            int endPage = sc.nextInt();

            return new Article(authors, title, magazineTitle, year, number, releaseYear, position, positionCount,
                    startPage, endPage);
        }
    }

    public static class PublicationComparator implements Comparator<Publication> {
        @Override
        public int compare(Publication a, Publication b) {
            if (a.getPoints() < b.getPoints()) {
                return 1;
            } else if (a.getPoints() > b.getPoints()) {
                return -1;
            }

            return 0;
        }
    }

    public static void main(String[] args) {
        String employee = sc.next();
        int publicationCount = sc.nextInt();
        Publication[] publications = new Publication[publicationCount];

        for (int i = 0; i < publicationCount; i++) {
            String publicationType = sc.next();

            publications[i] = PublicationFactory.getPublication(publicationType, employee);
        }

        Arrays.sort(publications, new PublicationComparator());

        for (int i = 0; i < publicationCount; i++) {
            Publication publication = publications[i];

            System.out.println(publication.toString());
        }

        sc.close();
    }
}
