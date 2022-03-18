**[Leetcode 96 不同的BST - Medium](https://leetcode-cn.com/problems/unique-binary-search-trees/)**

```java
class Solution {
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        //dp[i] = f(1) + f(2) + ... + f(n);f(i)为以i为根节点的BST个数
        //dp[i] = dp[0] * dp[n - 1] + dp[1] * dp[n - 2] + ... + dp[n - 1] * dp[0]
        //f(i)左子树个数为i - 1,右子树个数为n - i
        //f(i) = dp[i - 1] * dp[n - i];
        //先求dp[i],再递推出dp[n];
        for(int i = 2; i <= n; ++i) {
            for(int j = 1; j <= i; ++j) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }
}
```

> [题解](https://leetcode-cn.com/problems/unique-binary-search-trees/solution/hua-jie-suan-fa-96-bu-tong-de-er-cha-sou-suo-shu-b/)
>
> 动态规划，不过需要先预先处理出递推公式。





**[剑指Offer 10 - I fib - Easy](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)**

```java
class Solution {
    private static final int MOD = 1000000007;
    
    public int fib(int n) {
        if(n < 2) return n;
        int pre2 = 0, pre1 = 1, res = 1;
        for(int i = 2; i <= n; ++i) {
            res = (pre2 % MOD + pre1 % MOD) % MOD;
            pre2 = pre1;
            pre1 = res;
        }
        return res;
    }
}
```

> 数论处理大数越界





**[剑指Offer 10 - II 跳台阶 - Easy](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)**

```java
class Solution {
    private static final int MOD = 1000000007;
    
    public int numWays(int n) {
        if(n == 0) return 1;
        if(n == 1) return 1;
        int pre2 = 1, pre1 = 1, res = 1;
        for(int i = 2; i <= n; ++i) {
            res = (pre2 % MOD + pre1 % MOD) % MOD;
            pre2 = pre1;
            pre1 = res;
        }
        return res;
    }
}
```

> 同上



