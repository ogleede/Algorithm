## 最近最少使用页面置换算法

**[剑指Offer 146 LRU cache - Medium](https://leetcode-cn.com/problems/lru-cache/)**

```java
class LRUCache {
    private Node head = new Node(), tail = new Node();
    private int size = 0;
    private Map<Integer, Node> map;
    
    class Node {
        Node pre, next;
        int val, key;
        public Node() {}
        public Node(int val, int key) {this.val = val; this.key = key;}
    }
    
    public void deleteNode(Node node) {
        node.pre.next = node.next;
        node.next.pre = node.pre;
        node.next = null;
        node.pre = null;
    }
    
    public void addToHead(Node node) {
        Node sec = head.next;
        head.next = node;
        node.pre = head;
        node.next = sec;
        sec.pre = node;
    }
    
    public void moveToHead(Node node) {
        deleteNode(node);
        addToHead(node);
    }
    
    public LRUCache(int capacity) {
        size = capacity;
        map = new HashMap<>();
        head.next = tail;
        tail.pre = head;
    }
    
    public int get(int key) {
        if(!map.containsKey(key)) return -1;
        Node node = map.get(key);
        moveToHead(node);
        return node.val;
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)) {
            Node node = map.get(key);
            node.val = value;
            map.put(key, node);
            moveToHead(node);
        }else {
            if(map.size() == size) {
                Node node = tail.pre;
                deleteNode(node);
                map.remove(node.key);
            }
            Node node = new Node(value, key);
            map.put(key, node);
            addToHead(node);
        }
    }
}
```

> 希望get, put, delete, ordered。
>
> 哈希表可以满足前三项O(1)，但是无序
>
> 由于需要区别出不同页面的顺序，需要有序，链表有序。可以往头插
>
> 结合Hash和双向链表，可以实现整体O(1)