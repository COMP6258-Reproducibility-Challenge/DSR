# DSR
Basic re-implementation of Deep Symbolic Regression

# Dependencies
The project will make use of several Python packages, including:
- certifi==2022.12.7
- charset-normalizer==2.1.1
- cloudpickle==2.2.1
- Farama-Notifications==0.0.4
- filelock==3.9.0
- gymnasium==0.28.1
- idna==3.4
- importlib-metadata==6.3.0
- jax-jumpy==1.0.0
- Jinja2==3.1.2
- MarkupSafe==2.1.2
- mpmath==1.2.1
- networkx==3.0
- numpy==1.24.2
- Pillow==9.3.0
- requests==2.28.1
- scipy==1.10.1
- sympy==1.11.1
- torch==2.0.0
- typing_extensions==4.5.0
- urllib3==1.26.13
- zipp==3.15.0
To install the dependencies, run the following command:

`pip install -r requirements.txt`

# Usage
To run the project, simply run `python main.py`.

# Options
To change the target expression, create a tree using the add node function e.g. for x^3 + x^2 + x + 5:
```
expression.add_node(0)
expression.add_node(0)
expression.add_node(0)
expression.add_node(2)
expression.add_node(2)
expression.add_node(8)
expression.add_node(8)
expression.add_node(8)
expression.add_node(2)
expression.add_node(8)
expression.add_node(8)
expression.add_node(8)
expression.add_node(10)
expression.node_list[-1].set_value(5.0)
```
The index of each symbol is dictated by the library which we have already created an instance of. You can also include a Y or Const.
```
nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X]
nodes_list = [Add, Sub, Mult, Div, Sin, Cos, Log, Exp, X, Y, Const]
```


The dataset dictates the bounds of the expression and the number of datapoints.
```
dataset = Dataset(target_expr, numpoints=20, lb=-1, ub=1)
```

The different policy gradients are dictated by the loss function being used which has each loss function commented out:
```
loss = VPGLoss()
loss = RSPGLoss()
loss = PQTLoss(model, library, device=device)
```