```c++
std::unordered_map<T1, T1> my_map;
my_map[v1] = v2; // insert value
// find/insert average O(1), worst O(N)
```

```c++
std::map<T1, T1> my_map; // internally is binary search tree
my_map[v1] = v2; // insert value
// find/insert average O(N*logN)
my_map.begin()->first // get key of smallest element
my_map.rbegin()->first // get key of largest element
my_map.erase()
```

## Binary Tree
If it is binary search tree, then in-order traversal is left->mid->right. The output will be in ascending order.
```c++
/**
 * Binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */

```

## Binary Search Tree (C++ set)
```c++
set<T>
*set.begin()
*set.rbegin()
// find the first value that does not go below val
*set.lower_bound(val)
// find the first value that does goes above val
*set.upper_bound(val)
set.insert()
set.find()
set.erase()
```

## Hash table (c++ unordered_set)
```c++
unordered_set<T>
unordered_set.insert()
unordered_set.find()
unordered_set.erase()
```

## Double Linked List
```c++
list<T> l;
l.begin();
l.front();
l.back();
l.push_front(val);
l.push_back(val);
l.insert(iterator, val);
l.sort();
l.size();
l.erase(iterator);
```


## Min Stack
Store the min at push time for each node.
```c++
class MinStack {
public:
    MinStack() {
        stack.clear();
    }
    
    void push(int val) {
        int min_val = val;
        if (!stack.empty()) {
            min_val = min(stack[stack.size()-1].second, min_val);
        }
        stack.push_back({val, min_val});
    }
    
    void pop() {
        stack.pop_back();
    }
    
    int top() {
        return stack[stack.size()-1].first;
    }
    
    int getMin() {
        return stack[stack.size()-1].second;
    }
    std::vector<std::pair<int, int>> stack;
};
```

## LRU Cache
Could be implemented by linked list plus hash map.

## Stack
Could be implemented by just array.
```c++
stack<int> stack;
stack.push(21);
stack.pop();
stack.top();
```

## Queue
Could be implemented by 2 stacks.

## Double Ended Queue
Vector of fixed size vecrors.
```c++
deque<T> q;
T val = q[i];
```
- Random access - constant O(1) 
- Insertion or removal of elements at the end or beginning - constant O(1) 
- Insertion or removal of elements - linear O(n)


## TreeMap (JAVA) or Map (C++)
It is basically red-black tree / binary search tree. Gurantees O(logN) time cost for the containsKey, get, put and remove operations.
```c++
for (auto const& [key, val] : map) {

}
```

## HashMap (JAVA) or UnorderedMap (C++)
Hash map.

## Heap / Max Heap / Binary Heap
Max heap is a binary tree where the root is always greater than or equal to the children. The time complexities are \
Operation |    Avg  |  Worst \
Search |	O(n) |	O(n) \
Insert |	O(1) |	O(log n) \
Find-min |	O(1) |	O(1)

## Priority Queue
Priority Queue is similar to queue where we insert an element from the back and remove an element from front, but with a one difference that the logical order of elements in the priority queue depends on the priority of the elements. Note that it only has strict weak ordering. The element with highest priority will be moved to the front of the queue and one with lowest priority will move to the back of the queue. Thus it is possible that when you enqueue an element at the back in the queue, it can move to front because of its highest priority. \
Could be implemented by a max heap

```c++
priority_queue<T> pq;
pq.top(); // yeah, not front, top!
// by default, it sort from big to small.... which is the opposite comparing to other containers. And with greater, it sorts from small to large. For other containers, its also the opposite.!!!
priority_queue<T, vector<T>, greter<T>>;
```

## Union Find / Disjoint-set
It stores a partition of a set into disjoint subsets. It provides operations for adding new sets, merging sets (replacing them by their union), and finding a representative member of a set. The last operation makes it possible to find out efficiently if any two elements are in the same or different sets.

Time
- find: avg log*N
- merge ave log*N
```c++
// if group_id = id, then its the group representative
// otherwise it stroes an upper level element in its group
int group_ids[N] = {1,2,...N};
// size of each group, only valid at the representative pos
int gourp_sizes[N] = {1,1,...};
void merge(int group_ids[], int group_sizes[], int g1, int g2) {
    if (g1 == g2) return;
    // merge small group to bigger group
    if (group_sizes[g1] < group_sizes[g2]) {
        swap(g1, g2);
    }
    group_ids[g2] = g1;
    group_sizes[g1] += group_sizes[g2];
}
int find(int group_ids[], int id) {
    // if its not the representative of a group
    while(group_ids[id] != id) {
        // compress the path by 1 step
        group_ids[id] = group_ids[group_ids[id]];
        id = group_ids[id];
    }
    return id;
}
int betterFind(int group_ids[], int id) {
    // if its not the representative of a group
    int root = id;
    while(group_ids[root] != root) {
        root = group_ids[root];
    }
    // path compression
    while(group_ids[id] != root) {
        int tmp = group_ids[id];
        group_ids[id] = root;
        id = tmp;
    }
    return root;
}
```

## Trie
```c++
struct TrieNode {
    TrieNode *next[26] = {};
    int cnt = 0;
};
// or
struct TrieNode {
    TrieNode *next[26] = {};
    bool is_end = 0;
};
```

## C++ Function - equal_range(.begin(), .end(), val)
Returns std::pair<itr, itr> for the bound [start, end) that contains the values equals to val.
The elements in the range shall already be sorted according to this same criterion (operator< or comp), or at least partitioned with respect to val.

## K-d tree
Use for range search, nearest neighbor search etc.
