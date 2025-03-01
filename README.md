# ARIA  (AI research ideas archive)


## 1. Renforced Steering Vectors
There is no good way to increase thinking time. (no adding wait token is not a good solution).
steering vectors good ideas except. training SAE is expensive. two token diff steering vector don’t work reliably. (eg sv = '</think>' - '<think>'  model start to output garbage).

we can use rl to find steering vectors.  this will be usefull as you can find steering vectors that have deeper meaning of thinking. instead of just steering to wait token. or away </think>. 
we dont know all tokens model using it to pivot thinking we should limit it from only what we know. 

more uses cases, 
- suppose your cot have capalities of tool use. you can control model thinking for more or less tool use. 
- specilly useful for agents.
- its even useful for non reasoner llm, model that have tool use. make it take less, output struted data. other ways you can use control vectors.

todo:
- find sv for increasing length of output for r1 distill models

#### Results
did not work. rl is to noisy to find good steering vectors. 
https://x.com/shxf0072/status/1895817735260815475

todo: move to graveyard

## 2. Tool use inside cot
let model use tools inside cot. 
todo:
- build env for code tool use 
- find dataset for it
- you do batched generation, find some way to do this 
- tool use should need structured output need cold start data for it


## 3. Expectation Guided Tree Search
The expectation value of the known answer sequence $Y$ can be determined by analyzing the probability distributions of generated tokens. We define the approach as follows:

what dose this do?
for given know input x and answer y. find cot t that lead to y.

### Algorithm:
1. **Generate $n$ Samples**
   Each sample consists of a sequence of tokens:
```math
   x, t_{11}, t_{12}, t_{13}, t_{14}, \dots \\
   x, t_{21}, t_{22}, t_{23}, t_{24}, \dots \\
   x, t_{n1}, t_{n2}, t_{n3}, t_{n4}, \dots
```
   where $x$ is the initial input, and each $t_{ij}$ represents a token in the sequence.
2. **Stop at a Random Token**
   For each sequence, we stop at a random token position $k$:
```math
   x, t_{i1}, t_{i2}, \dots, t_{ik}
```
   where $i$ represents the sequence number and $k$ is the stopping position.

3. **Append the Known Answer**
   We concatenate the known answer sequence $Y$ to each truncated sample:
```math
   x, t_{11}, t_{12}, \dots, t_{1k}, Y \\
   x, t_{21}, t_{22}, \dots, t_{2k}, Y \\
   \vdots \\
   x, t_{n1}, t_{n2}, \dots, t_{nk}, Y
```

4. **Compute Logits for the Answer Sequence**
   The model generates logits for the known answer sequence, allowing us to compute the expectation value:
```math
   E[Y \mid t_{1:k}] = \sum_{i=1}^{|Y|} P(y_i \mid x, t_{1:k})
```
   where:
   - $Y$ is the known answer sequence
   - $y_i$ is the $i^{th}$ token in $Y$
   - $P(y_i \mid x, t_{1:k})$ is the probability of token $y_i$ given the context
   - $t_{1:k}$ represents the sequence up to position $k$

5. **Select Top-$k$ Chains Based on Expectation Values**
   We retain only the top-$k$ sequences with the highest expectation values:
   
```math
   (x, t_{11}, t_{12}, \dots, t_{1k}), E[Y \mid t_{1:k}] \\
   (x, t_{21}, t_{22}, \dots, t_{2k}), E[Y \mid t_{2:k}] \\
   \vdots \\
   (x, t_{n1}, t_{n2}, \dots, t_{nk}), E[Y \mid t_{n:k}]
```
   that maximize the expectation over $Y$. This helps identify the optimal sequence of tokens that lead to $Y$ without requiring brute-force search over an enormous space.

### Benefits:
- **Efficient Guided Search:** Instead of generating an enormous number of samples $N$ to find $Y$, this method allows a subset of $n$ samples to be used in a guided search.
- **Gradient-Free Optimization:** The approach provides a way to optimize for the answer sequence $Y$ without requiring backpropagation or differentiable loss functions.
- **Tree Search Enhancement:** The method can be incorporated into tree search algorithms to improve sampling efficiency and guide the search towards high-probability paths.

This technique ensures that even for models with limited capacity, the probability space can be efficiently explored to locate the correct answer without exhaustive sampling.
