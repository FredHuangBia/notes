# ML System Design - CS239s by Stanford

## Model offline evaluation
- Simpson's paradox
  - So sliced or fine-grained evaluation is important

- Model calibration
  - Platt scaling

- Model confidence measurement

- Enlarge batch size with limited mem
  - gradient checkpointing
  - gradient accumulation

- Model performance is not the same as business performance

- Out of distribution detection
  - Energy based model

- Types of tests
  - Perturbation test
    - Inject noise to test
  - Invariance test
    - Model output should be invariant to some input changes
  - Directional test
    - Model output should go towards some direction given certain changes in input
  - 

## Distributed Training
- Data parallel
  - Async
    - Parameter server
  - Synced
    - Mirrored
  - Usually could use larger learning rate when effective batch size is larger, because gradient will be more stable and less noisy

- Model parallel

## Framming the probelm smartly
Say if we want to predict what app the user is most likely to open next. There are two approaches
- Method 1
  - Input: User feature, Environment context
  - Output: A vector of N scores where N is number of apps
  - Pro: Only run once for all apps
  - Con: Need to retrain whenever a new app is added/removed!
- Method 2
  - Input: User feature, Environment context, App feature
  - Output: A score for each app
  - Pro: No need to retrain when new app is added/removed
  - Con: Need to run once for each app

## Decoupling the objectives
Say if we are designing a model to feed users with content. We want to make sure the content we feed is both **high quality**(not spam, not misinfo, not hated speach etc) and **high engagement**(user will more likely click it). We have 2 methods
- Method 1
  - Train a model optimize a combined loss, such as $\alpha L_{quality} + \beta L_{engagement}$

- Method 2
  - Train two models separately, on the 2 losses respectively.

We would favor method 2 because
- Its easier to train, optimizing one objective is easier than 2
- Easier to tweak, whenever we want to change $\alpha$ and $\beta$, method 1 requires a retrain, while method 2 does not.
- Easier to maintain. Different objectives might need different maintenance schedules, such as one evolves/changes faster than the other etc.

## OLTP vs OLAP


## Batch Prediction vs. Online Prediction
#### Batch Prediction - Asynchronous Prediction
Model training is actually batch processing. General use cases are video or shopping recommendation system, where you only need to update your recommendation ever hour or so.
#### Online Prediction - Synchronous Prediction
Face ID, speech recognition etc. **Having two different pipelines to process your data for batch (training) and online (deploy) is a common cause for bugs in ML production.** 

## Model on cloud vs model on edge
There are a lot of benefits to run models on edge devices
- Can run without internet
- Lower latency (no internet transaction time)
- Cheaper! The company dont need to pay the cloud comping fee
- More secure for sensitive user data

## Model Compression
Model compression is techniques that makes the model smaller without sacrifice its accuracy, so that it uses less memory and runs faster. Common techniques are
- Low Rank Factorization, such as depth-wise convolution.
- Knowledge Distillation, aka train a smaller network with supervision from a larger network, not used widly in production.
- Prunning. Prunning has two meanings, one is to remove entire nodes of a network, which changes the architechure. The other is set the value of some less significant parameters to 0, which makes the model sparse. Its interesting that there's paper showed that the large sparse model after pruning outperformed the retrained small counterpart.
- Quantization. Need to be careful about the range of the values. And be careful some small values may be rounded to 0. There's also QAT, which allows you to train larger model on the same hardware.

## Model Compiling
Intermediate representation (IR) lie at the core of how compilers work. From the original code for a model, compilers generate a series of high- and low-level intermediate representations before generating the code native to a hardware backend so that it can run on that hardware backend. This process is also called “lowering”, as in you “lower” your high-level framework code into low-level hardware-native code.

## Model Optimization
Once you lowerd your model representation, you could start optimize the model for better performance/efficiency on dedicated hardware. There are two ways to optimize your ML models: locally and globally. Locally is when you optimize an operator or a set of operators of your model. Globally is when you optimize the entire computation graph end-to-end.

Local optimizations:
- vectorization
- parallelization
- loop tiling: leverage the storage pattern to increase cache hit
- operator fusion: remove redudant intermediate memory access

Global optimizations:
- Graph optimization (tensorrt)
  - Vertical fusion
  - Horizontal fusion

Its also possible to use ML to optimize ML

## ML in Browser
Convert to WebAssembly(WASM). Faster than JavaScript but still slow.

## Feature Engineering
### Handle missing value
- Deletion
  - **Row deletion** if small portion of the data entries are missing this value
  - **Column deletion** if a big portion of the data entries are missing this value
- Imputaion
- Scaling
  - Normalization $x'=\frac{x-min(x)}{max(x)-min(x)}$
  - Standardization with scaling $x'=\frac{x-mean(x)}{std(x)}$
  - **Log transform** works well with skewed data distribution

### Discretization
Discretization is bucketing continuous feature into a few categorical buckets. Such as range of salary, range of age. The benefit is instead of learning continuous features, the algorithm only needs to learn to predict on a few categories. To define meanful boundaries, you can try plotting the histograms of the values.

### Encoding Categorical Features
Sometimes there might be infinite many categories, or continuously new categories. For example, new brands on amazon. Some times, even infinite many categories, for example, the user account or email address. How do we deal with this without always retraining our model whenever a new category comes out? There's a very smart but hacky way: **the hashing trick**. Basically we hash the feature, and the hash value will be the category index now. 

One problem with hased functions is collision, but we could enlarge the hash space, or use multiple hash functions to avoid it. And the impact of collision in real world is small. You could also choose locality-sensitive hashing, where similar categories are hased into close by values.

### Feature crossing
Feature crossing can be used to combine multiple (usually 2) categorical features into one. Basically, the space of the new feature is just the cross product of the combined features. It is useful to model non-linear relationships between features for linear models. Deep learning usually dont need this trick. There are some caveats of feature crossing, such as feature space blow up, and overfitting due the the blowed up feature space.

## Data leakage
Data leakage happends when information not availabel during test scenario is injected in training set. Common data leakage sources are preprocessing, future data, etc.