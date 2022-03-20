### LC

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



### 剑指

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







### 股票问题

**[股票问题通解](https://leetcode-cn.com/circle/article/qiAgHn/)**



**[Leetcode 121 买卖股票的最佳时机 - Easy](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)**

```java
T[i][1][0] = max(T[i - 1][1][0], T[i - 1][1][1] + prices[i])
T[i][1][1] = max(T[i - 1][1][1], T[i - 1][0][0] - prices[i]) = max(T[i - 1][1][1], -prices[i])
```



```java
//套模板 31ms 5%
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for(int i = 1; i < len; ++i) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], -prices[i]);
        }
        return dp[len - 1][0];
    }
}

//空间降维,3ms 30%
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int p0 = 0, p1 = -prices[0];
        for(int i = 1; i < len; ++i) {
            p0 = Math.max(p0, p1 + prices[i]);
            p1 = Math.max(p1, -prices[i]);
        }
        return p0;
    }
}


```







**[Leetcode 122 买卖股票的最佳时机 II - Medium](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)**

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for(int i = 1; i < len; ++i) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[len - 1][0];
    }
}

class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int p0 = 0, p1 = -prices[0];
        for(int i = 1; i < len; ++i) {
            int preP0 = p0, preP1 = p1;
            p0 = Math.max(p0, p1 + prices[i]);
            p1 = Math.max(preP1, preP0 - prices[i]);
        }
        return p0;
    }
}
```







**[Leetcode 123 买卖股票的最佳时机 III - Hard](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)**

```java
//73ms 19%
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int[][][] dp = new int[len][3][2];
        dp[0][1][0] = 0;
        dp[0][1][1] = -prices[0];
        dp[0][2][1] = -prices[0];//***
        dp[0][2][0] = 0;//***
        for(int i = 1; i < len; ++i) {
            dp[i][1][0] = Math.max(dp[i - 1][1][0], dp[i - 1][1][1] + prices[i]);
            dp[i][2][0] = Math.max(dp[i - 1][2][0], dp[i - 1][2][1] + prices[i]);
            dp[i][1][1] = Math.max(dp[i - 1][1][1], -prices[i]);
            dp[i][2][1] = Math.max(dp[i - 1][2][1], dp[i - 1][1][0] - prices[i]);
        }
        return dp[len - 1][2][0];//***
    }
}

//1ms 100%
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int p10 = 0, p11 = -prices[0], p20 = 0, p21 = -prices[0];
        for(int i = 1; i < len; ++i) {
            int preP10 = p10, preP11 = p11, preP20 = p20, preP21 = p21;
            p10 = Math.max(preP10, preP11 + prices[i]);
            p20 = Math.max(preP20, preP21 + prices[i]);
            p11 = Math.max(preP11, -prices[i]);
            p21 = Math.max(preP21, preP10 - prices[i]);
        }
        return p20;//***
    }
}
```





**[Leetcode 188 买卖股票的最佳时机 IV - Hard](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)**

```java
class Solution {
    public int maxProfit(int k, int[] prices) {
        if(prices == null || prices.length == 0 || k < 1) return 0;
        int len = prices.length;
        if(k >= len / 2) return maxProfit(prices);
        int[][][] dp = new int[len][k + 1][2];
        for(int i = 1; i <= k; ++i) {
            dp[0][i][0] = 0;
            dp[0][i][1] = -prices[0];
        }
        for(int i = 1; i < len; ++i) {
            for(int j = 1; j <= k; ++j) {
                dp[i][j][0] = Math.max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i]);
            }
        }
        return dp[len - 1][k][0];
    }
    
    private int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int p0 = 0, p1 = -prices[0];
        for(int i = 1; i < len; ++i) {
            int preP0 = p0, preP1 = p1;
            p0 = Math.max(p0, p1 + prices[i]);
            p1 = Math.max(preP1, preP0 - prices[i]);
        }
        return p0;
    }
}
```





**[Leetcode 309 最佳买卖股票时机含冷冻期 - Medium](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)**

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for(int i = 1; i < len; ++i) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], (i >= 2 ? dp[i - 2][0] : 0) - prices[i]);
        }
        return dp[len - 1][0];
    }
}


class Solution {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int preP0 = 0, p0 = 0, p1 = -prices[0];
        for(int i = 1; i < len; ++i) {
            int oldPreP0 = preP0, oldP0 = p0, oldP1 = p1;
            p0 = Math.max(oldP0, oldP1 + prices[i]);
            p1 = Math.max(oldP1, (i >= 2 ? oldPreP0 : 0) - prices[i]);
            preP0 = oldP0;
        }
        return p0;
    }
}
```





**[Leetcode 714 买卖股票的最佳时机含手续费 - Medium](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)**

```java
//卖出扣费
class Solution {
    public int maxProfit(int[] prices, int fee) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        for(int i = 1; i < len; ++i) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        }
        return dp[len - 1][0];
    }
}

//买入扣费
class Solution {
    public int maxProfit(int[] prices, int fee) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int[][] dp = new int[len][2];
        dp[0][0] = 0;
        dp[0][1] = -prices[0] - fee;
        for(int i = 1; i < len; ++i) {
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i] - fee);
        }
        return dp[len - 1][0];
    }
}

class Solution {
    public int maxProfit(int[] prices, int fee) {
        if(prices == null || prices.length == 0) return 0;
        int len = prices.length;
        int p0 = 0, p1 = -prices[0] - fee;
        for(int i = 1; i < len; ++i) {
            int preP0 = p0, preP1 = p1;
            p0 = Math.max(preP0, preP1 + prices[i]);
            p1 = Math.max(preP1, preP0 - prices[i] - fee);
        }
        return p0;
    }
}
```







