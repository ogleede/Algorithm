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





**[Leetcode 337打家劫舍III - Medium](https://leetcode-cn.com/problems/house-robber-iii/)**

```java
class Solution {
    private int[] dfs(TreeNode root) {
        if(root == null) return new int[]{0,0};
        int[] left = dfs(root.left);
        int[] right = dfs(root.right);
        //dp[0] 表示当前node不偷，dp[1] 表示当前node偷
        int[] dp = new int[2];
        dp[0] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        dp[1] = root.val + left[0] + right[0];
        return dp;
    }
    
    public int rob(TreeNode root) {
        int[] res = dfs(root);
        return Math.max(res[0], res[1]);
    }
}
```

> 树形dp入门题，也和股票类型问题类似，通过增加一维来消除后效性。
>
> [题解](https://leetcode-cn.com/problems/house-robber-iii/solution/shu-xing-dp-ru-men-wen-ti-by-liweiwei1419/)





**[]()**

**[Leetcode 413 等差数列划分 - Medium](https://leetcode-cn.com/problems/arithmetic-slices/)**

```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int len = nums.length;
        if(len < 3) return 0;
        int[] dp = new int[len];
        for(int i = 2; i < len; ++i) {
            if(nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                dp[i] = dp[i - 1] + 1;
            }
        }
        int res = 0;
        for(int num : dp) {
            res += num;
        }
        return res;
    }
}
```





**[Leetcode 542 01矩阵 - Medium](https://leetcode-cn.com/problems/01-matrix/)**

```java
//左上右下两次dp选最小值
//一开始先给1节点赋最大值
class Solution {
    public int[][] updateMatrix(int[][] mat) {
        int r = mat.length, c = mat[0].length;
        int[][] dp = new int[r][c];
        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < c; ++j) {
                if(mat[i][j] == 1) dp[i][j] = 10001;
            }
        }
        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < c; ++j) {
                if(i > 0) dp[i][j] = Math.min(dp[i][j], dp[i - 1][j] + 1);
                if(j > 0) dp[i][j] = Math.min(dp[i][j], dp[i][j - 1] + 1);
            }
        }
        for(int i = r - 1; i >= 0; --i) {
            for(int j = c - 1; j >= 0; --j) {
                if(i < r - 1) dp[i][j] = Math.min(dp[i][j], dp[i + 1][j] + 1);
                if(j < c - 1) dp[i][j] = Math.min(dp[i][j], dp[i][j + 1] + 1);
            }
        }
        return dp;
    }
}

//想分别从左上右下积累，并求两次最小值，并未通过
class Solution {
    public int[][] updateMatrix(int[][] mat) {
        int r = mat.length, c = mat[0].length;
        int[][] dp = new int[r][c];
        int[][] dp1 = new int[r][c];
        int[][] dp2 = new int[r][c];
        for(int i = 1; i < r; ++i) {
            if(mat[i][0] == 0) continue;
            dp1[i][0] = dp1[i - 1][0] + 1;
        }
        for(int j = 1; j < c; ++j) {
            if(mat[0][j] == 0) continue;
            dp1[0][j] = dp1[0][j - 1] + 1;
        }
        for(int i = 1; i < r; ++i) {
            for(int j = 1; j < c; ++j) {
                if(mat[i][j] == 0) continue;
                dp1[i][j] = Math.min(dp1[i - 1][j], dp[i][j - 1]) + 1;
            }
        }
        for(int i = r - 2; i >= 0; --i) {
            if(mat[i][c - 1] == 0) continue;
            dp2[i][c - 1] = dp2[i + 1][c - 1] + 1;
        }
        for(int j = c - 2; j >= 0; --j) {
            if(mat[r - 1][j] == 0) continue;
            dp2[r - 1][j] = dp2[r - 1][j + 1] + 1;
        }
        for(int i = r - 2; i >= 0; --i) {
            for(int j = c - 2; j >= 0; --j) {
                if(mat[i][j] == 0) continue;
                dp2[i][j] = Math.min(dp2[i + 1][j], dp2[i][j + 1]) + 1;
            }
        }
        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < c; ++j) {
                if(mat[i][j] == 0) continue;
                dp[i][j] = Math.min(dp1[i][j], dp2[i][j]);
            }
        }
        dp[0][0] = dp2[0][0];
        dp[r - 1][c - 1] = dp1[r - 1][c - 1];
        return dp;
    }
}

//多源BFS，图的BFS必须标记是否访问过，树是有方向的则不用标记
//假设多源有一个共通的超级源头
class Solution {
    private static int[][] dirs = {{1, 0},{-1, 0},{0, 1},{0, -1}};
    
    private boolean inArea(int x, int y, int r, int c) {
        return x >= 0 && x < r && y >= 0 && y < c;
    }
    
    public int[][] updateMatrix(int[][] mat) {
        int r = mat.length, c = mat[0].length;
        Queue<int[]> queue = new LinkedList<>();
        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < c; ++j) {
                if(mat[i][j] == 0) {
                    queue.offer(new int[]{i, j});
                }else {
                    mat[i][j] = -1;
                }
            }
        }
        while(!queue.isEmpty()) {
            int[] poll = queue.poll();
            int x = poll[0], y = poll[1];
            for(int[] dir : dirs) {
                int nx = x + dir[0], ny = y + dir[1];
                if(inArea(nx, ny, r, c) && mat[nx][ny] == -1) {
                    queue.offer(new int[]{nx, ny});
                    mat[nx][ny] = mat[x][y] + 1;
                }
            }
        }
        return mat;
    }
}
```







**[Leetcode 221 最大正方形 - Medium](https://leetcode-cn.com/problems/maximal-square/)**

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        int r = matrix.length, c = matrix[0].length;
        int[][] dp = new int[r + 1][c + 1];
        int max = 0;
        for(int i = 1; i <= r; ++i) {
            for(int j = 1; j <= c; ++j) {
                if(matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        return max * max;
    }
}
```

> 题解
> 对于在矩阵内搜索正方形或长方形的题型，一种常见的做法是定义一个二维 dp 数组，其中 
> dp[i][j] 表示满足题目条件的、以 (i, j) 为右下角的正方形或者长方形的属性。对于本题，则表示 
> 以(i, j) 为右下角的全由1 构成的最大正方形面积。如果当前位置是0，那么dp[i][j] 即为0；如果 
> 当前位置是1，我们假设dp[i][j] = k 2，其充分条件为dp[i-1][j-1]、dp[i][j-1] 和dp[i-1][j] 的值必须 
> 都不小于( k − 1)2，否则(i, j) 位置不可以构成一个面积为 k 2 的正方形。同理，如果这三个值中的 
> 的最小值为 ( k − 1)2，则 (i, j) 位置一定且最大可以构成一个面积为 k 2 的正方形。





**[Leetcode 279 完全平方数 - Medium](https://leetcode-cn.com/problems/perfect-squares/)**

```java
class Solution {
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for(int i = 1; i <= n; ++i) {
            for(int j = 1; j * j <= i; ++j) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }
}
```

> 妙，
>
> 分割类问题，状态转移一般不依赖与相邻位置。这题依赖于i - k^2位置







**[Leetcode 91 解码方法 - Medium](https://leetcode-cn.com/problems/decode-ways/)**

```java
class Solution {
    public int numDecodings(String s) {
        if(s == null || s.length() == 0) return 0;
        int len = s.length();//数据长度
        s = "_" + s;//加哨兵，避免讨论边界
        char[] cs = s.toCharArray();
        int[] dp = new int[len + 1];
        dp[0] = 1;//成功的初始条件
        for(int i = 1; i <= len; ++i) {
            int a = cs[i] - '0', b = (cs[i - 1] - '0') * 10 + a;
            if(a > 0 && a < 10) dp[i] = dp[i - 1];
            if(b > 9 && b < 27) dp[i] += dp[i - 2];
        }
        return dp[len];
    }
}
```







**[Leetcode 300 最长递增子序列 - Medium](https://leetcode-cn.com/problems/longest-increasing-subsequence/)**

```java
//O(n^2)
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        Arrays.fill(dp, 1);
        int res = 0;
        for(int i = 0; i < len; ++i) {
            for(int j = 0; j < i; ++j) {
                if(nums[i] > nums[j]) dp[i] = Math.max(dp[i], dp[j] + 1);
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
}

//O(nlogn)
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if(len < 2) return len;
        int[] tail = new int[len];
        tail[0] = nums[0];
        int end = 0;
        for(int i = 1; i < len; ++i) {
            if(nums[i] > tail[end]) tail[++end] = nums[i];
            else {
                int l = 0, r = end;//搜索插入位置
                while(l < r) {
                    int mid = l + (r - l) / 2;
                    if(tail[mid] < nums[i]) l = mid + 1;
                    else r = mid;
                }
                tail[l] = nums[i];
            }
        }
        return end + 1;
    }
}
```

