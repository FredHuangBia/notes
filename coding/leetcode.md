## Q1: Two sum
```
Naive solution
O(N^2)
Sort the array, then use 2 pointers
O(NlogN)
Use unordered_map
O(N)
```

## Q3: Longest Substring Without Repeating Characters
```
Two pointers + map O(N)
Between p1 and p2 is the current substring
Use map to store last seen position of each char. We move p2 to the right as extending new char. If a new char's last seen index is smaller than p1 or it does not exist in the map, then it is not repeating. Otherwise, increase p1 to 1 char after the previous appearance of the new char.
```

## * Q5: Longest Palindromic Substring
```
Method1: DP
Create look up table P(i,j) which is true of substring from i to j is palindrom. Then P(i,j) = P(i+1,j-1) && S(i)==S(j)
Method2: Expand from center
There are 2n-1 centers (some centers are between 2 letters)
```

## Q11: Container With Most Water
```
Two Pointers
Start from the widest container. For a shorter container to have bigger volume, it must have a larger lowest height. So If left is lower, then increament left pointer, else decrement right pointer.
```

## * Q15: 3Sum
``` c++
// Difficulty is remove duplicate. To avoid reusing the same number, we make sure indexs i < j < k. To avoid duplicate result, we sort the array and skip if nums[i] == nums[i-1] or nums[j] == nums[j-1]. For speedup, we use a map to store val to its lartest index.
vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> results;
    sort(nums.begin(), nums.end());
    // map from value to its largest index
    unordered_map<int, int> my_map;
    for (int i = 0; i < nums.size(); ++i) my_map[nums[i]] = i;
    for (int i = 0; i < nums.size(); ++i) {
        // If nums[i] == nums[i-1], then at nums[i], we already added
        // all possible values of nums[i] and nums[i+]. If we do it
        // again for nums[i+1], there'll be duplication.
        if (i && nums[i] == nums[i-1]) continue;
        // j > i
        for (int j = i+1; j < nums.size(); ++j) {
            if (j > i+1 && nums[j] == nums[j-1]) continue;
            int res = 0 - nums[i] - nums[j];
            if (my_map.find(res) == my_map.end()) continue;
            // make sure k > j
            if (my_map[res] > j) results.push_back({nums[i], nums[j], res});
        }
    }
    return results;
}
```

## * Q16: 3Sum closest
```
Sort the array. Then for each index i, use 2 pointers j=i+1, k=len-1, and increament j if sum is smaller, decrement k if sum is larger.
```

## * Q18: 4Sum / kSum
``` c++
// Sort the array. Use 2 pointers inner loop, and k-2 outer loops
vector<vector<int>> fourSum(vector<int>& nums, int target) {
    vector<vector<int>> results;
    sort(nums.begin(), nums.end());
    for(int p1 = 0; p1 < nums.size(); ++p1) {
        if (p1 > 0 && nums[p1] == nums[p1-1]) continue;
        for(int p2 = p1+1; p2 < nums.size(); ++p2) {
            if (p2 > p1+1 && nums[p2] == nums[p2-1]) continue;
            int two_sum = nums[p1] + nums[p2];
            if (two_sum > target/2) continue; // this line is optional
            int p3 = p2+1;
            int p4 = nums.size() - 1;
            while (p3 < p4) {
                if (p3 > p2+1 && nums[p3] == nums[p3-1]) {
                    ++p3;
                    continue;
                }
                if (p4 < nums.size()-1 && nums[p4] == nums[p4+1]) {
                    --p4;
                    continue;
                }
                long four_sum = static_cast<long>(two_sum) + static_cast<long>(nums[p3]) + static_cast<long>(nums[p4]);
                if (four_sum == target) {
                    results.push_back({nums[p1], nums[p2], nums[p3], nums[p4]});
                    ++p3;
                    --p4;
                } else if (four_sum < target) {
                    ++p3;
                } else {
                    --p4;
                }
            }
        }
    }
    return results;
}
```

## Q19: Remove Nth Node From End of List
Maintain two pointers and update one with a delay of n steps.
```c++
ListNode* removeNthFromEnd(ListNode* head, int n) {
    int dist = 0;
    ListNode* current = head;
    ListNode* node_before_delete = head;
    while (current->next != nullptr) {
        current = current->next;
        if (dist >= n) {
            node_before_delete = node_before_delete->next;
        } else {
            ++dist;
        }
    }
    // deleting head
    if (dist == n-1) {
        return node_before_delete->next;
    } else {
        ListNode* next_node = n == 1 ? nullptr : node_before_delete->next->next;
        node_before_delete->next = next_node;
        return head;
    }
}
```

## * Q22: Generate Parentheses
Recurssion. And to avoid repeat, we only either append a "(" or ")" to the left of the sub strings. To make sure string is valid, we keep a counter for remaining number of left and right parentheses, and only add right parentheses if remaining right is more than left. 
```c++
class Solution {
public:
    vector<string> generateParenthesisImpl(int left, int right) {
        vector<string> combs;
        if (left == 0 && right == 0) return combs;
        if (left == 0 && right == 1) {
            combs.push_back(")");
            return combs;
        }
        if (left > 0) {
            vector<string> sub_combs = generateParenthesisImpl(left-1, right);
            for (string comb : sub_combs) combs.push_back("("+comb);
        }
        if (right > left) {
            vector<string> sub_combs = generateParenthesisImpl(left, right-1);
            for (string comb : sub_combs) combs.push_back(")"+comb);
        }
        return combs;
    }
    vector<string> generateParenthesis(int n) {
        vector<string> combs = generateParenthesisImpl(n, n);
        return combs;
    }
};
```

## * 39. Combination Sum
Given a non-repeat list of values, return all possible combinations that sum up to target. 
DFS!

## 45. Jump Game II
Given a list of jump range at each position, return minimum number of jumps it take to get to last node. BFS + table to stroe explored nodes.
```c++
void jumpImpl(const vector<int>& nums, vector<int>& min_jumps, queue<int>& to_visits) {
    while (to_visits.size() > 0) {
        int num_to_visits = to_visits.size();
        for (int pos_idx = 0; pos_idx < num_to_visits; ++pos_idx) {
            int pos = to_visits.front();
            to_visits.pop();
            int jump_range = nums[pos];
            int num_jump = min_jumps[pos];
            for (int i = 1; i <= jump_range; ++i) {
                int new_pos = pos + i;
                if (new_pos >= nums.size()) return;
                if (min_jumps[new_pos] < 0) {
                    min_jumps[new_pos] = num_jump + 1;
                    to_visits.push(new_pos);
                }
            }            
        }            
    }
}
int jump(vector<int>& nums) {
    vector<int> min_jumps(nums.size(), -1);
    min_jumps[0] = 0;
    queue<int> to_visits;
    to_visits.push(0);
    jumpImpl(nums, min_jumps, to_visits);
    return min_jumps[nums.size()-1];
}
```

## 49. Group Anagrams
Ex:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
Sort every string, use a hash map to store sorted string to result group index map.

## 50. Pow(x, n)
Recurssion, pow(x, n) = pow(x, n/2)^2 or pow(x, n/2)^2*x;

## 55. Jump Game
[2,3,1,1,4], return whether we can jump to end
```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        if (nums.size() <= 1) return true;
        int max_reach = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (max_reach < i) return false;
            max_reach = max(max_reach, i+nums[i]);
        }
        return max_reach >= nums.size()-1;
    }
};
```

## 81. * Search in Rotated Sorted Array II
sorted repeated array rotated at a pivot, search if a value exists.

无重复的 rotated sorted list 的情况下可以通过比较找到严格递增的一边，就可以判断出pivot在哪边，然后就可以知道该搜左边还是右边。但这个不行，有一种可能是rotate在了重复的数字上，导致无法判断pivot在左边还是右边，这时候就得都搜。比如[1,0,1,1,1], [1,5,1,1,1] 都是可能的。
```c++
class Solution {
public:
    bool searchImpl(vector<int>& nums, int target, int low, int high) {
        if (low > high) return false;
        int mid = (low + high) / 2;
        if (nums[mid] == target) return true;
        if (nums[mid] > nums[low]) {
            // left must be in ascending order
            if (target < nums[mid] && target >= nums[low]) {
                return searchImpl(nums, target, low, mid-1);
            } else {
                return searchImpl(nums, target, mid+1, high);
            }
        } else if (nums[mid] < nums[high]) {
            // right must be in ascending order
            if (target <= nums[high] && target > nums[mid]) {
                return searchImpl(nums, target, mid+1, high);
            } else {
                return searchImpl(nums, target, low, mid-1);
            }
        }
        bool found = searchImpl(nums, target, low, mid-1) || searchImpl(nums, target, mid+1, high);
        return found;
    }
    bool search(vector<int>& nums, int target) {
        int low = 0;
        int high = nums.size()-1;
        return searchImpl(nums, target, low, high);
    }
};
```

## 86. Partition List
Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
Make 2 lists, one conatin values smaller, the other conatin values larger. Then combine the two.
```c++
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* left_list = new ListNode(0);
        ListNode* right_list = new ListNode(0);
        ListNode* left_head = left_list;
        ListNode* right_head = right_list;
        while(head != nullptr){
            if(head->val < x){
                left_list->next = head;
                left_list = left_list->next;
                head = head->next;
                left_list->next = nullptr;
            }
            else{
                right_list->next = head;
                right_list = right_list->next;
                head = head->next;
                right_list->next = nullptr;
            }
        }
        left_list->next = right_head->next;
        return left_head->next;
    }
};
```

## * 94. Binary Tree Inorder Traversal
Inorder: left, mid, right
Iterative sol is difficult.
```c++
void inorderTraversalRecurssion(TreeNode* root, vector<int>& results) {
    if (root == nullptr) return;
    if (root->left != nullptr) inorderTraversalRecurssion(root->left, results);
    results.push_back(root->val);
    if (root->right != nullptr) inorderTraversalRecurssion(root->right, results);
}

vector<int> inorderTraversalIterative(TreeNode* root) {
    vector<int> values;
    stack<TreeNode*> my_stack;
    TreeNode* node = root;
    while (node != nullptr || !my_stack.empty()) {
        if (node != nullptr) {
            my_stack.push(node);
            node = node->left;
        } else {
            node = my_stack.top();
            my_stack.pop();
            values.push_back(node->val);
            node = node->right;
        }
    }
    return values;
}
```

## * 95. Unique Binary Search Trees II
Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n.
Sol: Recurssive, make every node root, then find all comb of lefts (values < root) and rights (values > root).
```c++
vector<TreeNode*> generateTrees(int start, int end) {
    if (start > end) return {nullptr};
    if (start == end) return {new TreeNode(start)};
    vector<TreeNode*> results;
    for (int i = start; i <= end; ++i) {
        vector<TreeNode*> lefts = generateTrees(start, i-1);
        vector<TreeNode*> rights = generateTrees(i+1, end);
        for (auto left : lefts) {
            for (auto right : rights) {
                TreeNode* root = new TreeNode(i);
                root->left = left;
                root->right = right;
                results.push_back(root);
            }
        }
    }
    return results;
}

vector<TreeNode*> generateTrees(int n) {
    return generateTrees(1, n);
}
```

## * 97. Interleaving String
```c++
// 2D DP
bool isInterleave(string s1, string s2, string s3) {
    if (s1.size() + s2.size() != s3.size()) return false;
    if (s1.size() == s3.size()) return s1 == s3;
    if (s2.size() == s3.size()) return s2 == s3;
    vector<vector<bool>> dp(s1.size()+1, vector<bool>(s2.size()+1, false));
    for (int i = 0; i <= s1.size(); ++i) {
        for (int j = 0; j <= s2.size(); ++j) {
            if (i == 0 && j == 0) dp[i][j] = true;
            else if (i == 0) dp[i][j] = dp[i][j-1] && s2[j-1] == s3[i+j-1];
            else if (j == 0) dp[i][j] = dp[i-1][j] && s1[i-1] == s3[i+j-1];
            else dp[i][j] = dp[i-1][j] && s1[i-1] == s3[i+j-1] || dp[i][j-1] && s2[j-1] == s3[i+j-1];
        }
    }
    return dp[s1.size()][s2.size()];

// 1D DP
bool isInterleave(string s1, string s2, string s3) {
    if (s1.size() + s2.size() != s3.size()) return false;
    if (s1.size() == s3.size()) return s1 == s3;
    if (s2.size() == s3.size()) return s2 == s3;
    vector<bool> dp(s2.size()+1, false);
    for (int i = 0; i <= s1.size(); ++i) {
        for (int j = 0; j <= s2.size(); ++j) {
            if (i == 0 && j == 0) dp[j] = true;
            else if (i == 0) dp[j] = dp[j-1] && s2[j-1] == s3[i+j-1];
            else if (j == 0) dp[j] = dp[j] && s1[i-1] == s3[i+j-1];
            else dp[j] = dp[j-1] && s2[j-1] == s3[i+j-1] || dp[j] && s1[i-1] == s3[i+j-1];
        }
    }
    return dp[s2.size()];
}
}
```

## * 99. Recover Binary Search Tree
You are given the root of a binary search tree (BST), where the values of exactly two nodes of the tree were swapped by mistake.
Sol: Use in-order traversal, usually the values should be in ascending order. However, if 2 nodes are swapped, then their order will be wrong. We just need to find the 2 places where the value is smaller than previous value in in order traversal, those two need to be swapped. But there are 2 situations, one is the 2 values are adjacent, the other is they are not adjacent. Such as [1,2,3,4] -> [1,4,3,4] or [1,3,2,4].
```c++
void recoverTreeImpl(TreeNode*& now, TreeNode*& prev, TreeNode*& to_swap_1, TreeNode*& to_swap_2) {
    if (now == nullptr) return;
    recoverTreeImpl(now->left, prev, to_swap_1, to_swap_2);
    if (prev != nullptr && now->val < prev->val) {
        if (to_swap_1 == nullptr) {
            to_swap_1 = prev;
        }
        if (to_swap_1 != nullptr) {
            to_swap_2 = now;
        }
    }
    prev = now;
    recoverTreeImpl(now->right, prev, to_swap_1, to_swap_2);
}

void recoverTree(TreeNode* root) {
    TreeNode* to_swap_1 = nullptr;
    TreeNode* to_swap_2 = nullptr;
    TreeNode* prev = nullptr;
    recoverTreeImpl(root, prev, to_swap_1, to_swap_2);
    int to_swap_1_val = to_swap_1->val;
    to_swap_1->val = to_swap_2->val;
    to_swap_2->val = to_swap_1_val;
}
```

## * 117. Populating Next Right Pointers in Each Node II
Given a binary tree, populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
Sol: use a helper temp node in each row makes impl simpler. Fill each row from the row above.

```c++
Node* connect(Node* root) {
    Node* now = root;
    while (now != nullptr) {
        Node* temp = new Node(0);
        Node* child = temp;
        while (now != nullptr) {
            if (now->left != nullptr) {
                child->next = now->left;
                child = child->next;
            }
            if (now->right != nullptr) {
                child->next = now->right;
                child = child->next;
            }
            now = now->next;
        }
        now = temp->next;
        delete temp;
    }
    return root;
}
```

## * 128.  Longest Consecutive Sequence
Use hash table. Only check for consequetive when num-1 does not exist, so that it is the begining of the streak.
```c++
int longestConsecutive(vector<int>& nums) {
    unordered_map<int, bool> my_map;
    for (int num : nums) {
        my_map[num] = true;
    }
    int longest_streak = 0;
    for (int num : nums) {
        if (my_map.find(num-1) == my_map.end()) {
            int current_streak = 1;
            while (my_map.find(++num) != my_map.end()) current_streak++;
            longest_streak = max(current_streak, longest_streak);
        }
    }
    return longest_streak;
}
```

## **146. LRU Cache
Use hash map + double linked list. C++ has a list<T> data structure that is actually double linked list. It supports O(1) insertion and deletion anywhere.
```c++
class LRUCache {
public:
        list<pair<int,int>> l;
        unordered_map<int,list<pair<int, int>>::iterator> m;
        int size;
        LRUCache(int capacity)
        {
            size=capacity;
        }
        int get(int key)
        {
            if(m.find(key)==m.end())
                return -1;
            l.splice(l.begin(),l,m[key]);
            return m[key]->second;
        }
        void put(int key, int value)
        {
            if(m.find(key)!=m.end())
            {
                l.splice(l.begin(),l,m[key]);
                m[key]->second=value;
                return;
            }
            if(l.size()==size)
            {
                auto d_key=l.back().first;
                l.pop_back();
                m.erase(d_key);
            }
            l.push_front({key,value});
            m[key]=l.begin();
        }
};
```

## **169. Majority Element
Given an array nums of size n, return the majority element (element appear > 2/n times). 
Sol: Boyer-Moore Voting Algorithm, O(N) time, O(1) space. Have a counter initialize as 0. Whenever the counter is 0, we assign the current value to majority. If current value is same as current majority, we +1, else -1. Say we start from a value, and reaches the first cnt=0. If the starting value is the result, then so far, we've disgarded same number of result and wrong values, so that cnter=0. If the starting value is not the result, then half of the values we disgarded must be this wrong value. No matter what, in the remaining of the list, the result is still majority. 
```c++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int cnt = 0;
        int result = nums[0];
        for (int i = 0; i < nums.size(); ++i) {
            if (cnt == 0) result = nums[i];
            if (nums[i] == result) {
                ++cnt;
            } else {
                --cnt;
            }
        }
        return result;
    }
};
```

## ** 189. Rotate Array
Given an array, rotate the array to the right by k steps, where k is non-negative.
Reverse first n-k elements, then reverse last k elements, finally reverse all.

[1,2,3,4,5,6,7] rotate 3
step 1,2 [4,3,2,1,7,6,5]
step 3 [5,6,7,1,2,3,4]


## ** 209. Minimum Size Subarray Sum
Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.

Sol: Two pointers.


## ** 215. Kth Largest Element in an Array
Find Kth largest element in given array.
Sol: QuickSelect! O(N) time.

## ** 232. Implement Queue using Stacks
Amortized O(1) for all operations. Implement stacks with queue can't achieve this time complexity. Because if we transfer all elems from one queue to another, the order does not change, but if its stack, the order will be reversed.
```c++
class MyQueue {
public:
    MyQueue() {
        s1 = {};
        s2 = {};
    }
    
    void push(int x) {
        s1.push(x);
    }
    
    int pop() {
        peek();
        int val = s2.top();
        s2.pop();
        return val;
    }
    
    int peek() {
        if (s2.empty()) {
            while (!s1.empty()) {
                s2.push(s1.top());
                s1.pop();
            }
        }
        return s2.top();
    }
    
    bool empty() {
        return s1.empty() && s2.empty();
    }

private:
    stack<int> s1;
    stack<int> s2;
};
```

## ** 241. Different Ways to Add Parentheses
Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators.

Divide and conquer
```c++
class Solution {
public:
    unordered_map<string, vector<int>> mem;
    vector<int> diffWaysToCompute(string expression) {
        if (mem.find(expression) != mem.end()) return mem[expression];
        vector<int> left;
        vector<int> right;
        vector<int> results;
        for (int i = 0; i < expression.size(); ++i) {
            char c = expression[i];
            if (c == '+' || c == '-' || c == '*') {
                left = diffWaysToCompute(expression.substr(0, i));
                right = diffWaysToCompute(expression.substr(i+1, expression.size()-i-1));
            }
            for (int l : left) {
                for (int r : right) {
                    if (c == '+')
                        results.push_back(l + r);
                    else if (c == '*')
                        results.push_back(l * r);
                    else if (c == '-')
                        results.push_back(l - r);
                }
            }
        }
        if (results.empty()) {
            results.push_back(stoi(expression));
        }
        mem[expression] = results;
        return results;
    }
};
```

## 311. Sparse Matrix Multiplication
To show off...

```c++
template <typename T>
class SparseMatrix {
 public:
  SparseMatrix(const vector<vector<T>>& mat, bool row_major = true) {
      row_maj = row_major;
      m = mat.size();
      n = mat[0].size();
      for (int r = 0; r < mat.size(); ++r) {
          for (int c = 0; c < mat[0].size(); ++c) {
              if (mat[r][c] != 0) {
                  row_maj ? data[r].push_back({mat[r][c], c}) : data[c].push_back({mat[r][c], r});
              }
          }
      }
  }
 
 // TODO(fred): make them private and write access function
  unordered_map<int, vector<pair<T, int>>> data;
  bool row_maj = true;
  int m;
  int n;
};

template <typename T>
vector<vector<T>> operator*(SparseMatrix<T>& m1, SparseMatrix<T>& m2) {
 if (!m1.row_maj) {
     throw std::invalid_argument("m1 must be row major");
 }
 if (m2.row_maj) {
     throw std::invalid_argument("m2 must be col major");
 } 
 vector<vector<T>> result(m1.m, vector<T>(m2.n, 0));
 for (int r = 0; r < m1.m; ++r) {
     if (m1.data.find(r) == m1.data.end()) continue;
     for (int c = 0; c < m2.n; ++c) {
         if (m2.data.find(c) == m2.data.end()) continue;
         T sop = 0;
         vector<pair<T, int>>& row = m1.data[r];
         vector<pair<T, int>>& col = m2.data[c];
         int p1 = 0;
         int p2 = 0;
         while (p1 < row.size() && p2 < col.size()) {
             if (row[p1].second == col[p2].second) {
                 sop += row[p1++].first * col[p2++].first;
             } else if (row[p1].second < col[p2].second) {
                 ++p1;
             } else {
                 ++p2;
             }
         }
         result[r][c] = sop;
     }
 }
 return move(result);
}

class Solution {
public:
    vector<vector<int>> multiply(vector<vector<int>>& mat1, vector<vector<int>>& mat2) {
        SparseMatrix smat1(mat1);
        SparseMatrix smat2(mat2, /*row_maj=*/false);
        return smat1 * smat2;
    }
};
```

## 489. Robot Room Cleaner
Backtrack + hash map for constrained programming
```c++
/**
 * // This is the robot's control interface.
 * // You should not implement it, or speculate about its implementation
 * class Robot {
 *   public:
 *     // Returns true if the cell in front is open and robot moves into the cell.
 *     // Returns false if the cell in front is blocked and robot stays in the current cell.
 *     bool move();
 *
 *     // Robot will stay in the same cell after calling turnLeft/turnRight.
 *     // Each turn will be 90 degrees.
 *     void turnLeft();
 *     void turnRight();
 *
 *     // Clean the current cell.
 *     void clean();
 * };
 */

class Solution {
public:
    void backtrack(Robot& robot, pair<int, int> pos, int dir) {
        string pos_str = to_string(pos.first) + "," + to_string(pos.second);
        cleaned.insert(pos_str);
        robot.clean();
        for (int i = 0; i < 4; ++i) {
            int new_dir = (i + dir) % 4;
            pair<int, int> new_pos = pos;
            new_pos.first += directions[new_dir].first;
            new_pos.second += directions[new_dir].second;
            string new_pos_str = to_string(new_pos.first) + "," + to_string(new_pos.second);
            if (cleaned.find(new_pos_str) == cleaned.end() && robot.move()) {
                backtrack(robot, new_pos, new_dir);
                // come back
                robot.turnRight();
                robot.turnRight();
                robot.move();
                robot.turnRight();
                robot.turnRight();
            }
            robot.turnRight();
        }
    }
    void cleanRoom(Robot& robot) {
        backtrack(robot, {0,0}, 0);
        return;
    }
    
private:
    unordered_set<string> cleaned;
    vector<pair<int, int>> directions = {{-1,0},{0,1},{1,0},{0,-1}};
};
```

## 715. Range Module
A Range Module is a module that tracks ranges of numbers. Design a data structure to track the ranges represented as half-open intervals and query about them.

Sol: Use map to store <start, end> pairs. Use lower bound and upper bound, be very careful about corner conditions.
```c++
class RangeModule {
public:
    RangeModule() {
        lut = {{INT_MIN, INT_MIN}, {INT_MAX, INT_MAX}};
    }
    
    void addRange(int left, int right) {
        // cout << "Adding ---- \n";
        auto it_low = lut.lower_bound(left);
        if (it_low != lut.begin()) --it_low;
        if (it_low != lut.end() && it_low->second < left) ++it_low;
        auto it_high = lut.upper_bound(right);
        auto it_high_prev = prev(it_high);
        int new_low = left;
        int new_high = right;
        new_low = min(it_low->first, new_low);
        new_high = max(it_high_prev->second, new_high);
        for (auto it = it_low; it != it_high; ++it) {
            // cout << "Remove " << it->first << " " << it->second << "\n";
        }
        lut.erase(it_low, it_high);
        lut[new_low] = new_high;
        // cout << "Add " << new_low << ' ' << new_high << "\n";
    }
    
    bool queryRange(int left, int right) {
        auto it_low = lut.lower_bound(left); //first >= left
        if (it_low != lut.begin()) --it_low; // first < left
        if (it_low != lut.end() && it_low->second < left) ++it_low;
        return it_low->first <= left && it_low->second >= right;
    }
    
    void removeRange(int left, int right) {
        // cout << "Removing ---- \n";
        auto it_low = lut.lower_bound(left);
        if (it_low != lut.begin()) --it_low;
        if (it_low != lut.end() && it_low->second < left) ++it_low;
        auto it_high = lut.upper_bound(right);
        auto it_high_prev = prev(it_high);
        vector<vector<int>> to_add;
        if (left > it_low->first) {
            to_add.push_back({it_low->first, left});
        }
        if (right < it_high_prev->second && right >= it_high_prev->first) {
            to_add.push_back({right, it_high_prev->second});
        }
        for (auto it = it_low; it != it_high; ++it) {
            // cout << "Remove " << it->first << " " << it->second << "\n";
        }
        lut.erase(it_low, it_high);
        for (vector<int>& rg : to_add) {
            lut[rg[0]] = rg[1];
            // cout << "Add back " << rg[0] << ' ' << rg[1] << "\n";
        }
    }
private:
    map<int, int> lut;
};
```

## 778. Swim in Rising Water
Binary search + DFS. 
```c++
class Solution {
public:
    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size();
        bool visited[50][50] = {};
        memset(visited, false, sizeof(visited));
        int l = max({2*(n-1), grid[0][0], grid[n-1][n-1]});
        int r = n * n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (dfs(grid, visited, mid, {0,0})) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
            memset(visited, false, sizeof(visited));
        }
        return l;
    }
    bool dfs(const vector<vector<int>>& grid, bool visited[50][50], int t, vector<int> pos) {
        int n = grid.size();
        if (pos[0] < 0 || pos[1] < 0 || pos[0] >= n || pos[1] >= n || visited[pos[0]][pos[1]] || grid[pos[0]][pos[1]] > t) return false;
        visited[pos[0]][pos[1]] = true;
        if (pos[0] == n-1 && pos[1] == n-1) return true;
        for (vector<int> direction : directions) {
            vector<int> new_pos = pos;
            new_pos[0] += direction[0];
            new_pos[1] += direction[1];
            if (dfs(grid, visited, t, new_pos)) return true;
        }
        return false;
    }
private:
    vector<vector<int>> directions = {{1,0}, {0,1}, {-1,0}, {0,-1}};
};
```

## 818. Race Car
Return the length of the shortest sequence of instructions to get there.

Sol: Because it is shartest len, we think abnout BFS. 
```c++
// Its hard to prove the correctness of this method
class Solution {
public:
    int racecar(int target) {
        // BFS
        queue<vector<int>> my_q;
        my_q.push({0,0,1}); // step, pos, spd
        while (!my_q.empty()) {
            vector<int> state = my_q.front();
            my_q.pop();
            int step = state[0];
            int pos = state[1];
            int spd = state[2];
            
            if (pos < 0 || pos > 2 * target) continue;
            if (pos == target) return step;
            
            // Try A
            my_q.push({step+1, pos+spd, 2*spd});
            // Try B
            // When we pass the target, there's no meaning to keep driving the same direction. And if we are driving aweay from target, we definitely want to consider coming back.
            if ((spd < 0 && pos + spd < target) || (spd > 0 && pos + spd > target))
                my_q.push({step+1, pos, spd>0?-1:1});
        }
        return 0;
    }
};

// Use a set<vector<int>>, where each vector is <pos,spd> to serve as a visited set. Because we don't know what speeds we will observe, so using a set insetad of dp.
class Solution {
public:
    int racecar(int target) {
        set<vector<int>> seen;
        queue<vector<int>> todo;
        todo.push({0, 1, 0}); // {pos, speed, steps}
        seen.insert({0, 1});
        while (!todo.empty()) {
            vector<int> cur = todo.front();
            todo.pop();
            int pos = cur[0];
            int speed = cur[1];
            int steps = cur[2];
            // If our position is target
            if (pos == target) {
                return steps;
            }
            // Try A
            if ((pos + speed <= 10000 && pos + speed > 0) && seen.find({pos + speed, speed * 2}) == seen.end()) {
                todo.push({pos + speed, speed * 2, steps + 1});
                seen.insert({pos + speed, speed * 2});
            }
            // Try R
            int newSpeedR = (speed > 0 ? -1 : 1);
            if (seen.find({pos, newSpeedR}) == seen.end()) {
                todo.push({pos, newSpeedR, steps + 1});
                seen.insert({pos, newSpeedR});
            }
        }
        return -1;
    }
};
```

## 1105. Filling Bookcase Shelves
Memo + dfs
```c++
class Solution {
public:
    int minHeightImpl(const vector<vector<int>>& books, int shelfWidth, int i, int w, int h, vector<vector<int>>& mem) {
        if (i >= books.size()) return h;
        if (mem[i][w] == 0) {
            // start a new row
            mem[i][w] = h + minHeightImpl(books, shelfWidth, i+1, books[i][0], books[i][1], mem);
            if (w + books[i][0] <= shelfWidth) {
                mem[i][w] = min(mem[i][w], minHeightImpl(books, shelfWidth, i+1, w+books[i][0], max(h, books[i][1]), mem));
            }
        }
        return mem[i][w];
    }
    int minHeightShelves(vector<vector<int>>& books, int shelfWidth) {
        vector<vector<int>> mem(books.size()+1, vector<int>(shelfWidth+1, 0));
        return minHeightImpl(books, shelfWidth, 0, 0, 0, mem);
    }
};
```

## 1293. Shortest Path in a Grid with Obstacles Elimination
Shortest path, so we use BFS. For speedup, we use dp, to store a 2d array, the value is number of breakers reamining last time we visit this node. If the current number of breakers is smaller than last time, we dont need to revist, because last time we gurantee to be a smaller step, and if more breakers, last visit is strictly better. An speed up trick is using A* heuristic search. If the breakers is more than the manhatan distance to dest, we could directly jump to dest, no need to exploer more.
```c++
class Solution {
public:
    void bfs(priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>>& to_explore, vector<vector<int>>& explored, const vector<vector<int>>& grid, int& steps) {
        while (!to_explore.empty()) {
            vector<int> node = to_explore.top();
            to_explore.pop();
            int i = node[1];
            int j = node[2];
            int k = node[3];
            int s = node[4];
            if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size()) continue;
            if (explored[i][j] >= k) continue;
            k = k - grid[i][j];
            if (k < 0) continue;
            explored[i][j] = k;
            if (i == grid.size()-1 && j == grid[0].size()-1) {
                steps = s;
                return;
            }
            if (k >= grid.size()-1-i + grid[0].size()-1-j) {
                steps = s + grid.size()-1-i + grid[0].size()-1-j;
                return;
            }
            
            to_explore.push({s-i-j-1, i+1, j, k, s+1});
            to_explore.push({s-i-j-1, i, j+1, k, s+1});
            to_explore.push({s-i-j+1, i-1, j, k, s+1});
            to_explore.push({s-i-j+1, i, j-1, k, s+1});
            
        }
        return;
    }
    
    int shortestPath(vector<vector<int>>& grid, int k) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> explored(m, vector<int>(n, -1));
        priority_queue<vector<int>, vector<vector<int>>, greater<vector<int>>> to_explore;
        to_explore.push({0, 0, 0, k, 0});
        int steps = -1;
        bfs(to_explore, explored, grid, steps);
        return steps;
    }   
```

## 1610. Maximum Number of Visible Points
1. convert x,y to polar angles with atan2
2. sort the ploar angles
3. duplicate the polar angles by +360 degrees, this is the key! for example, 0 and 270 actually only spans 90 instead of 270.
4. Use 2 pointers to go through the sorted duplicated list, to find maximum number of visible points
```c++
constexpr float PI = 3.141592653579;

class Solution {
public:
    int visiblePoints(vector<vector<int>>& points, int angle, vector<int>& location) {
        int result = 0;
        vector<double> polar_angles;
        for (vector<int> pt : points) {
            int x = pt[0] - location[0];
            int y = pt[1] - location[1];
            if (x == 0 && y == 0) {
                result++;
            } else {
                double angle = atan2((double)y, (double)x) * 180.0 / PI;
                polar_angles.push_back(angle);
            }
        }
        sort(polar_angles.begin(), polar_angles.end());
        int size = polar_angles.size();
        for (int i = 0; i < size; ++i) {
            polar_angles.push_back(360 + polar_angles[i]);
        }
        int p1 = 0;
        int p2 = 0;
        int max_num = 0;
        while (p2 < polar_angles.size() && p1 <= p2) {
            double diff = polar_angles[p2] - polar_angles[p1];
            if (diff <= (double)angle + 1e-5) {
                max_num = max(max_num, p2-p1+1);
                ++p2;
            } else {
                ++p1;
            }
        }
        return max_num + result;
    }
};
```

## 1937. Maximum Number of Points with Cost
DP. For every row, first go from left to right, then go from right to left. O(MN) time.
```c++
class Solution {
public:
    long long maxPoints(vector<vector<int>>& points) {
        int m = points.size();
        int n = points[0].size();
        long long result;
        vector<long long> last_row(n);
        for (int i = 0; i < n; ++i) {
            last_row[i] = points[0][i];
        }
        for (int i = 1; i < m; ++i) {
            vector<long long> this_row(n, 0);
            // left to right
            int last_max_j = 0;
            long long last_max = last_row[0];
            this_row[0] = last_max + points[i][0];
            for (int j = 1; j < n; ++j) {
                if (last_row[j] >= last_max - j + last_max_j) {
                    last_max_j = j;
                    last_max = last_row[j];
                    this_row[j] = last_row[j] + points[i][j];
                } else {
                    this_row[j] = last_max - j + last_max_j + points[i][j];
                }
            }
            // right to left
            last_max_j = n-1;
            last_max = last_row[n-1];
            this_row[n-1] = max(this_row[n-1], last_max + points[i][n-1]);
            for (int j = n-2; j >= 0; --j) {
                if (last_row[j] >= last_max + j - last_max_j) {
                    last_max_j = j;
                    last_max = last_row[j];
                    this_row[j] = max(this_row[j], last_row[j] + points[i][j]);
                } else {
                    this_row[j] = max(this_row[j], last_max + j - last_max_j + points[i][j]);
                }
            }
            last_row = this_row;
        }
        long long max_val = last_row[0];
        for (int i = 1; i < n; ++i) {
            max_val = max(max_val, last_row[i]);
        }
        return max_val;
    }
};
```

## 2115. Find All Possible Recipes from Given Supplies
Tolological sort
```c++
vector<string> findAllRecipes(vector<string>& rec, vector<vector<string>>& ing, vector<string>& sup) {
        unordered_map<string,vector<string>> graph;
        int n = rec.size();
        unordered_set<string> s;
        for(auto x : sup) s.insert(x);            //store all the supplies in unordered set
		
        unordered_map<string,int> indegree;   //to store the indegree of all recipes
        for(auto x : rec)indegree[x] = 0;                      //initially take the indegree of all recipes to be 0
    
        for(int i = 0; i < n; i++){
            for(int j = 0; j < (int)ing[i].size(); j++){
                if(s.find(ing[i][j]) == s.end()){     
                    graph[ing[i][j]].push_back(rec[i]);    //if the ingredient required to make a recipe is not in supplies then  
                    indegree[rec[i]]++;                     //we need to make a directed edge from that ingredient to recipe
                }
            }
        }
        
        //KAHN'S ALGORITHM
        queue<string> q;
        for(auto x : indegree){
            if(x.second == 0){
                q.push(x.first);
            }
        }
       vector<string> ans;
        while(!q.empty()){
            string tmp = q.front();
            q.pop();
            ans.push_back(tmp);
            for(auto nbr : graph[tmp]){
                indegree[nbr]--;
                if(indegree[nbr] == 0)
                    q.push(nbr);
            }
        }
        return ans;
    }
```

## 2158. Amount of New Area Painted Each Day
Use a ordered map to store all the unpainted ranges. Then for every new paint, find all overlaping unpainted ranges. Count the new paint, and replace them with newly painted region.
```c++
class Solution {
public:
    int paintInterval(map<int, int>& wall, const vector<int>& interval) {
        int start = interval[0];
        int end = interval[1];
        auto it_low = wall.lower_bound(start);
        auto it_up = wall.upper_bound(end);
        if (it_low != wall.begin()) it_low--;
        if (it_low->second < start) it_low++;
        int count = 0;
        int it_low_start = it_low->first;
        int it_up_end = prev(it_up)->second;
        for (auto it = it_low; it != it_up; ++it) {
            count += min(it->second, end) - max(it->first, start);
        }
        wall.erase(it_low, it_up);
        if (it_low_start < start) {
            wall[it_low_start] = start;
        }
        if (it_up_end > end) {
            wall[end] = it_up_end;
        }
        return count;
    }
    vector<int> amountPainted(vector<vector<int>>& paint) {
        // intervals of unpainted wall
        map<int, int> wall;
        int wall_start = INT_MAX;
        int wall_end = INT_MIN;
        for (const auto& p : paint) {
            wall_start = min(wall_start, p[0]);
            wall_end = max(wall_end, p[1]);
        }
        wall[wall_start] = wall_end;
        wall[INT_MAX] = INT_MAX;
        wall[INT_MIN] = INT_MIN;
        vector<int> results;
        for (const auto& p : paint) {
            results.push_back(paintInterval(wall, p));
        }
        return results;
    }
};
```

## 2172. Maximum AND Sum of Array
Use bit mask and dp. Bitmask means which elemetns are selected. A trick is append 0s to nums to make it size 2 * numSlots. Because 0 & any slot is 0, it could be used to occupy a slot and represent it is empty.
```c++
class Solution {
public:
    int maximumANDSum(vector<int>& nums, int numSlots) {
        // append 0s to nums
        int N = 2 * numSlots;
        nums.resize(N);
        vector<int> dp(1 << N, 0);
        // i is a mask representing which nums are used
        for (int i = 1; i < 1 << N; ++i) {
            int cnt = __builtin_popcount(i);
            // the last slot we need to fill
            int slot = (cnt + 1) / 2;
            for (int j = 0; j < N; ++j) {
                // not using this value
                if ((i & (1<<j)) == 0) continue;
                // try put this value in the last slot
                dp[i] = max(dp[i], dp[i ^ (1<<j)] + (slot & nums[j]));
            }
        }
        return dp[(1 << N) - 1];
    }
};
```

## 2188. Minimum Time to Finish the Race
First figure out the min times without change tire. Then, 
dp[i] = min(changeTime + dp[i-n] + dp[n]) for n = [0,i-1]

```c++
class Solution {
public:
    int minimumFinishTime(vector<vector<int>>& tires, int changeTime, int numLaps) {
        // dp[i] = min(changeTime + dp[i-n] + dp[n]) for n = [1,i]
        vector<long long> dp(numLaps+1, INT_MAX);
        dp[0] = 0;
        for (int t = 0; t < tires.size(); ++t) {
            // assume not changing tire
            long long time = tires[t][0];
            long long last_lap_time = time;
            for (int l = 1; l <= min(20, numLaps); ++l) {
                dp[l] = min(dp[l], time);
                if (time > 1e6) break;
                last_lap_time *= tires[t][1];
                time = time + last_lap_time;
            }
        }
        for (int l = 2; l <= numLaps; ++l) {
            for (int j = 1; j < l; ++j) {
                dp[l] = min(dp[l], changeTime + dp[j] + dp[l-j]);
            }
        }
        return dp[numLaps];
    }
};
```

## 2421. Number of Good Paths
Union-find. The key is to reconstruct the tree/graph from low value nodes to large value nodes, and compute the number of new value nodes in each group. We need a sorted map to store value to node ids. And need a unordered map to store edges pointing from larger node to smaller node.
```c++
class Solution {
public:
    int find(int groups[], int p) {
        int grp = p;
        while (groups[grp] != grp) {
            grp = groups[grp];
        }
        while (groups[p] != p) {
            int tmp = groups[p];
            groups[p] = grp;
            p = tmp;
        }
        return grp;
    }
    
    void merge(int groups[], int grp_sizes[], int p, int q) {
        int p_grp = find(groups, p);
        int q_grp = find(groups, q);
        if (p_grp == q_grp) return;
        if (grp_sizes[p_grp] > grp_sizes[q_grp]) {
            swap(p_grp, q_grp);
        }
        groups[p_grp] = q_grp;
        grp_sizes[q_grp] += grp_sizes[p_grp];
        return;
    }

    int numberOfGoodPaths(vector<int>& vals, vector<vector<int>>& edges) {
        map<int, vector<int>> val2nodes;
        for (int node = 0; node < vals.size(); ++node) {
            int val = vals[node];
            val2nodes[val].push_back(node);
        }
        
        int groups[vals.size()];
        int grp_sizes[vals.size()];
        for (int i = 0; i < vals.size(); ++i) {
            groups[i] = i;
            grp_sizes[i] = 1;
        }
        // edges from large val to smaller or equal val, break tie by node id
        unordered_map<int, vector<int>> node2nodes;
        for (const vector<int>& edge : edges) {
            int node1 = edge[0];
            int node2 = edge[1];
            vector<int> val_node1 = {vals[node1], node1};
            vector<int> val_node2 = {vals[node2], node2};
            if (val_node1 <= val_node2) node2nodes[node2].push_back(node1);
            else node2nodes[node1].push_back(node2);
        }
        
        int result = 0;
        for (auto const& [val, nodes] : val2nodes) {
            result += nodes.size();
            unordered_map<int, int> grp2cnt;
            for (int node : nodes) {
                for (int small_node : node2nodes[node]) {
                    merge(groups, grp_sizes, node, small_node);
                }
            }
            for (int node : nodes) grp2cnt[find(groups, node)]++;
            for (const auto& [grp, cnt] : grp2cnt) {
                result += (cnt * (cnt-1)) / 2;
            }
        }
        return result;
    }
};
```