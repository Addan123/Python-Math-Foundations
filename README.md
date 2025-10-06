Python-Math-Foundations
Basic functions of python and math's

Python Practice: NumPy, Pandas, Probability, Linear Algebra & Linear Regression

This project is a **practice guide** for learning essential Python tools used in data science and machine learning. It demonstrates the use of **NumPy**, **Pandas**, **Probability concepts**, **Linear Algebra operations**, and implements **Linear Regression from scratch**.

---

 📦 Requirements

Make sure you have Python installed. Then install the libraries:

```bash
pip install numpy pandas
```

---

 📖 What You’ll Learn

1. **NumPy**

   * Arrays, shapes, mean, dot product, norm.
2. **Pandas**

   * Create DataFrames, inspect data, descriptive statistics.
3. **Probability**

   * Random numbers, mean, variance, standard deviation.
4. **Linear Algebra**

   * Dot products, transpose, matrix inverse.
5. **Linear Regression (from scratch)**

   * Using the **Normal Equation** to compute regression parameters.

---

 ▶️ How to Run

Run the script with:

```bash
python practice.py
```

It will print out results for each section:

* NumPy array operations
* Pandas DataFrame handling
* Probability statistics
* Linear Algebra examples
* Linear Regression coefficients (slope and intercept)

---

## 📝 Example Output

```
NumPy mean: 3.0 Dot: 55 Norm: 7.416198 Shape: (2, 3)
Pandas head:
     Name  Age  Score
0  Alice   25     85
1    Bob   30     90
Describe:
             Age      Score
count   2.000000   2.000000
mean   27.500000  87.500000
...
Prob mean: 0.123 Var: 0.97 Std: 0.984
A·v: [17 39] 
Transpose:
 [[1 3]
 [2 4]] 
Inverse:
 [[-2.   1. ]
 [ 1.5 -0.5]]
Regression theta: [2.01 2.95]
```

---

## 🚀 Next Steps

* Try modifying the dataset size in linear regression.
* Experiment with **Gradient Descent** instead of the Normal Equation.
* Explore more Pandas functions (`groupby`, `merge`, `pivot`).
* Dive deeper into probability distributions with `scipy.stats`.

---

Happy Learning! 🎯
