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