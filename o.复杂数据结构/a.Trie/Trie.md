**[剑指Offer II 62 实现前缀树 - Medium](https://leetcode-cn.com/problems/QC3q1f/)**

```java
class Trie {
    class TrieNode {
        boolean isEnd = false;
        TrieNode[] tns;
        public TrieNode() {
            tns = new TrieNode[26];
        }
    }
    
    private TrieNode root;
    
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        if(word == null || word.length() == 0) return;
        char[] cs = word.toCharArray();
        TrieNode p = root;
        for(char c : cs) {
            int idx = c - 'a';
            if(p.tns[idx] == null) p.tns[idx] = new TrieNode();
            p = p.tns[idx];
        }
        p.isEnd = true;
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        if(word == null || word.length() == 0) return false;
        char[] cs = word.toCharArray();
        TrieNode p = root;
        for(char c : cs) {
            int idx = c - 'a';
            if(p.tns[idx] == null) return false;
            p = p.tns[idx];
        }
        return p.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String word) {
        if(word == null || word.length() == 0) return false;
        char[] cs = word.toCharArray();
        TrieNode p = root;
        for(char c : cs) {
            int idx = c - 'a';
            if(p.tns[idx] == null) return false;
            p = p.tns[idx];
        }
        return true;
    }
}
```

> 用类的方式写出，后面在具体应用到其他题目的时候，可以灵活一些，不单独写成一个类。



**[剑指Offer II 63 替换单词 - Medium](https://leetcode-cn.com/problems/UhWRSj/)**

```java
class Solution {
    class Trie {
        class TrieNode {
            private boolean isEnd = false;
            private TrieNode[] tns;
            public TrieNode() {
                tns = new TrieNode[26];
            }
        }
        
        public Trie() {
            root = new TrieNode();
        }
        
        private TrieNode root;
    
        private void insert(String word) {
            if(word == null || word.length() == 0) return;
            char[] cs = word.toCharArray();
            TrieNode p = root;
            for(char c : cs) {
                int idx = c - 'a';
                if(p.tns[idx] == null) p.tns[idx] = new TrieNode();
                p = p.tns[idx];
            }
            p.isEnd = true;
        }
    
        private boolean hasPrefix(String word) {
            if(word == null || word.length() == 0) return false;
            char[] cs = word.toCharArray();
            TrieNode p = root;
            for(char c : cs) {
                if(p.isEnd) return true;//保证最短
                int idx = c - 'a';
                if(p.tns[idx] == null) return false;
                p = p.tns[idx];
            }
            return true;
        }
        
        private String cutToPrefix(String word) {
            StringBuilder tmp = new StringBuilder();
            TrieNode p = root;
            for(char c : word.toCharArray()) {
                int idx = c - 'a';
                if(p.isEnd || p.tns[idx] == null) break;
                tmp.append(c);
                p = p.tns[idx];
            }
            return tmp.toString();
        }
    }
    
    private Trie trie;
    
    public String replaceWords(List<String> dict, String sentence) {
        trie = new Trie();
        for(String word : dict) trie.insert(word);
        String[] words = sentence.split(" ");
        for(int i = 0; i < words.length; ++i) {
            if(trie.hasPrefix(words[i])) words[i] = trie.cutToPrefix(words[i]);
        }
        StringBuilder tmp = new StringBuilder();
        for(int i = 0; i < words.length - 1; ++i) {
            tmp.append(words[i]);
            tmp.append(" ");
        }
        tmp.append(words[words.length - 1]);
        return tmp.toString();
    }
}
```





**[剑指Offer II 64 神奇的字典 - Medium](https://leetcode-cn.com/problems/US1pGT/)**

```java
```







**[剑指Offer II 65 最短的单词编码 - Medium](https://leetcode-cn.com/problems/iSwD2y/)**

```java
class Solution {
    class TrieNode {
        private TrieNode[] tns;
        public TrieNode() {
            tns = new TrieNode[26];
        }
    }
    
    private TrieNode root;
    
    private int insert(String word) {
        TrieNode p = root;
        boolean isNew = false;
        char[] cs = word.toCharArray();
        for(int i = cs.length - 1; i >= 0; --i) {
            int idx = cs[i] - 'a';
            if(p.tns[idx] == null) {
                p.tns[idx] = new TrieNode();
                isNew = true;
            }
            p = p.tns[idx];
        }
        return isNew ? word.length() + 1 : 0;
    }
    
    public int minimumLengthEncoding(String[] words) {
        root = new TrieNode();
        Arrays.sort(words, (a, b) -> b.length() - a.length());
        int res = 0;
        for(String word : words) {
            res += insert(word);
        }
        return res;
    }
}
```







**[剑指Offer II 66 单词之和 - Medium](https://leetcode-cn.com/problems/z1R5dt/)**

```java
```





**[剑指Offer II 67 最大的异或 - Medium](https://leetcode-cn.com/problems/ms70jA/)**

```java
```

