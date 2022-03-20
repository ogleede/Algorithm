Scanner sc = new Scanner(System.in);

BufferedReader br = new BufferedReader(new inputStreamReader(System.in));

int n = Integer.parseInt(br.readLine().trim());

String[] strseq = br.readLine().trim().split(" ");









![截屏2022-03-18 下午4.59.27](Scanner.assets/%E6%88%AA%E5%B1%8F2022-03-18%20%E4%B8%8B%E5%8D%884.59.27.png)

```java
import java.util.*;
public class Main {
    public static void main(String[] args) {
        int m;
        double sum = 0, n;
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()) {
            n = sc.nextInt();
            m = sc.nextInt();
            for(int i = 0; i < m; ++i) {
                sum += n;
                n = Math.sqrt(n);
            }
            System.out.printf("%.2f", sum);
            System.out.println();
            
        }
    }
}
```







![截屏2022-03-18 下午5.00.03](Scanner.assets/%E6%88%AA%E5%B1%8F2022-03-18%20%E4%B8%8B%E5%8D%885.00.03.png)

```java
import java.util.*;
public class Main {
    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while(sc.hasNextInt()) {
            int m = sc.nextInt(), n = sc.nextInt();
            if(100 <= m && m <= n && n <= 999) {
                int flag = 0;
                for(int i = m; i <= n; ++i) {
                    int a0 = i % 10;
                    int a1 = i / 10 % 10;
                    int a2 = i / 100;
                    if(i == a0*a0*a0 + a1*a1*a1 + a2*a2*a2) {
                        ++flag;
                        if(flag > 1) System.out.print(" " + i);
                        else 		 System.out.print(i);
                    }
                }
                if(flag == 0) System.out.println("no");
            }
            System.out.println();
        }
    }
    
}
```





