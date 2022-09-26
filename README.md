# Amazon reviews
### Meet our team and project
We are Martí Cortada, Marçal Estruch, Jofre Poch and Pol Rifé, four enthusiast students in the world of Data Science and Engineering. We have created **Amazon_reviews** [public repository](https://github.com/MartiCortada/Amazon_reviews) in order to be used as a common workspace for the development of a very interesting project.

As far as the project concerns, in a nutshell, we have decided to apply a LSTM neural network architecture over the _Amazon_reviews_ dataset (which consists, as the name specifies, of a great amount of reviews from Amazon) in order to perform a text classification task. You will be able to get further information in the _Dataset Card_ and _Model Cart_ in the `Docs` folder of our repository.

### Repository architecture

* **Code** folder: it contains all the code of our project (it has folders inside whose filename is self-explanatory, e.g. _[Data_Acquisition_and_Understanding](https://github.com/MartiCortada/Amazon_reviews/tree/main/Code/Data_Acquisition_and_Understanding)_ of _[Deployment](https://github.com/MartiCortada/Amazon_reviews/tree/main/Code/Deployment)_ of the model. ).

* **Dataset** folder: it contains all the data needed for the project.

* **Docs** folder: it contains the [Dataset_Card.md](https://github.com/MartiCortada/Amazon_reviews/blob/main/Docs/Dataset/Dataset_Card.md) and [Model_Card.md](https://github.com/MartiCortada/Amazon_reviews/blob/main/Docs/Model/Model_Card.md) with specific information in each particular case. Please, consult it when needed to get further insight.

* .gitignore, README.md, requirements.txt and set_up.py files.

  These files are commonly well-known by the software engineering comunity. They are very useful when reproducing the work authors have done, keep track of library version control in case of using a specific environment when setting up. 

### How to reproduce our work?

Firstly, you should install or update our environment. Please, download the repository on your future local working directory (you may decide whether to use a virtual environment or not, it is up to you!). Then, type:

`python3 set_up.py`

This will set up the environment with the specific versions of packages we have used. In other words, it performs the following command:

`pip install -r requirements.txt`

After this, you are able to move through the folders with all requirements satisfied! 