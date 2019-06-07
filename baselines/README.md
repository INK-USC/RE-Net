# Code framework for KG completion.
- Code:
    - modified based on [knowledge_representation_pytorch](https://github.com/jimmywangheng/knowledge_representation_pytorch)
- Dataset:
    - 1st column in train.txt - subject entity
    - 2nd column - relation
    - 3rd column - object entity
    - 4th column - time

    - 1st figure in stat.txt - number of entities
    - 2nd figure in stat.txt - number of relations
    
    use `preprocess_TA_step1.py` and `preprocess_TA_step2.py` to make data for TATransE and TADistMult.
    ```
    python3 preprocess_TA_step1.py ICEWS18
    python3 preprocess_TA_step2.py ICEWS18
    ```  
    use `preprocess_TTransE.py` to make data for TTransE.  
    ```
    python3 preprocess_TTransE.py ICEWS18
    ```

- data.py: this is for corrupting triples and includes other functions for data

- util.py: this is a collection of frequent functions

- evaluation_modelX.py: evaluation codes

- TTransE.py, TATransE.py, TADistMult.py: train codes

- You can run the code with
	```
	python3 TTransE.py (-- parameters)
	python3 TATransE.py
	python3 TADistMult.py
	```
	e.g.:
	```
	CUDA_VISIBLE_DEVICES=0 python3 TTransE.py -f 1 -d ICEWS18 -L 1 -bs 1024 -n 1000
	```

