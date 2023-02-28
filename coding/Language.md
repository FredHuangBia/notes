# CUDA
## Function Execution Space Specifiers
- `__global__`: 
  - Runnable on device only
  - Callable on both device (latest version) and host
- `__device__`:
  - Runnable on device only
  - Callable on device only
- `__host__`:
  - Runnable on host only
  - Callable on host only

## Variable Memory Space Specifiers
- `__device__`: declares a variable that resides on the device, by default its on global memory space.
- `__constant__`: optionally used together with `__device__`, declares a variable that resides in constant memory space
- `__shared__`: optionally used together with `__device__`, declares a variable that resides in shared memory space

## Blocks
Each kernel could be launched with multiple blocks, the blocks could be up to 3 dimensional. We could access the block index by eg `blockIdx.x`, and the block dimensions by eg `blockDim.x`. 

## Threads
Each block could contain multiple threads. The threads could be up to 3 dimensional. And we could access the thread index by eg `threadIdx.x`. Usually, there are up to 1024 threads inside each block.

## Block shared mem and syn
Threads within a block can cooperate by sharing data through some shared memory. We can specify synchronization points in the kernel by calling `__syncthreads()`. This is useful for shared memory communication. For efficient cooperation, the shared memory is expected to be a low-latency memory near each processor core (much like an L1 cache) and `__syncthreads()` is expected to be lightweight.

## Memory Hierarchy
Per thread local registers -> Per block shared memory -> Distributed shared memory of all blocks in the cluster -> global memory of all clusters

There's also a read only texture memory.

## Unified memory
It is a coherant memory space shared between host and device that use the same address space. It is convinenet to be used to port GPU and CPU communication.

## Performance optimization
- Avoid memcpy, especially cross device.
- Avoid Control Flow Instructions.
- Dont do `__syncthreads()` too often.
- Minimize Memory Thrashing. Avoid malloc and free as much as you can.
- Use simple arithmatic operations
- Accessing the per block shared memory is faster than global

<br>

# C++
## What is OOP?
Procedural programming languages is about writing functions that process on data. But OOP allows you to create classes that contain both data and functions to process them.

## What is containership?
You can contain one class/struct in another.

## What is Encapsulation?
You can hide code or object member from other objects/callers/users etc to avoid unintentional changes or read.

## What do you understand about smart pointers in C++?
Smart pointers have auto garbage collection, to prevent memory leak. You dont need to manually free them.

## What is a singleton design pattern?
A class with a maximum of a single instance at any time. It cannot be instantiated further. This class provides a way to access its only object which can be accessed directly without need to instantiate the object of the class.

## What is `syncrhonized` qualifier?
It makes sure at most one thread could be accesing this function. It makes the function thread safe.

## What does the `static` qualifer do?
- **Static Variable in a Function** means a variable exists for the lifetime of the program. Even if the function is called multiple times, the space of that variable is always the same, and value is kept. It is useful to keep some state to the function.
- **Static Variable in a Class** is similar as above, it will be shared by all instances of the class. And static variable can't be initialized in class constructor.
- **Declare an object as static** means the object exists for the lifetime of the program, instead of a local scope.
- **Static function in a class** does not depend on instanciating the class. You dont need an instance of the class to call it, and as a result, it can only access other static variables or static functions of the class.

## What is the role of this pointer and void pointer?
**This pointer**: The 'this pointer' is present in the member functions of every object. It points to the object itself and can be used to access the object's data. 

**Void pointer**: A pointer that has no data type associated with it is called a void pointer. You can assign any type of pointer to a void pointer.

## What are virtual and pure virtual functions?
**Virtual function** is a functino that is implemented in the base class, and could be overridden in inherited class.

**Pure virtual function** is similar but it is not implemented in base class. Class with pure virtual function is abstract, and can not be instantiated. If the inherited class also did not implement the pure virtual function, then it is also an abstract class. It is declared by setting the fucntion to be `=0`.

Both virtual and pure virtual functions can not be static.

## What is a copy constructor?
A copy constructor is a member function that initializes the object with another object of the same class. It can be customized by the user like the following:
```c++
Point(const Point& p1) {}
```

## What's the difference between overloading and overriding?
Overloading is the same function name with different input type signatures. Overriding is the same function name and same input type signatures, but in parent-child classes.

## What is Polymorphism in C++?
Polymorphism in C++ means, the same entity (function or object) behaves differently in different scenarios. Function overloading, overridding, operator overridding, virtual functions etc. are all polymorphisms in C++.

## What are templates?
Template allows you to pass data type as a parameter to the function/class, so we dont need to write the same code for different class types. It is expanded at compile time.
```c++
template <typename T1, typename T2>
T1 myFunc(T1 a, T2 b) {}
```

## What are move, l value, and r value?
**l value** refers to the address that identifies an object, it is usually long-lasting. **r value** refers to the actual value stored at somewhere, it is short and temporary. Pointer is not l value, it is a type. L value only makes sense in an expression. Like `i=3`, the `i` is l value.

Just like copy constructor, **move** is another type of object consturctor. It is not moving anything, instead, it produces an rvalue reference to the object, it gives up the ownership of the being moved object, and transfers it to the assigned object. Using move in constructors can avoid copying/creating temp parameters. But we need to be careful, after move, the original object is no longer usable. For example, using a move constructor, a std::vector could just copy its internal pointer to data to the new object.

## What are different inheritances?
Private member is always inaccessible in inherited classes, but protected can. Both of them can not be accessed from outside the class.

**public inheritance**
- public -> public
- protected -> protected
- private -> inaccessible

**private inheritance**
- public -> private
- protected -> private
- private -> inaccessible

**protected inheritance**
- public -> protected
- protected -> protected
- private -> inaccessible


## Pass a function as parameter
```c++
int operation(int x, int y, std::function<int(int, int)> function){return function(x,y);}
```

## What is constexpr?
The constexpr specifier declares that it is possible to evaluate the value of the function or variable at compile time.

## Casting
Static Cast: This is the simplest type of cast which can be used. It is a compile time cast.It does things like implicit conversions between types (such as int to float, or pointer to void*), and it can also call explicit conversion functions (or implicit ones).

Dynamic casting is mainly used for safe downcasting at run time.
