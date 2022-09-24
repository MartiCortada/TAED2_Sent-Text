### LSTM (BASELINE)

Firstly, before giving an accurate description of our chosen model, let us introduce the model itself. So, our choice was to re-use a model which all of us were familiar with, but implementing some modifications in it. Hence, our decision was to use a [Long Short-Term Memory model (LSTM)](https://www.kaggle.com/code/polrifmata/lstm-baseline/notebook), which we had obtained from Kaggle repository. In fact, the entire code model was created by [José A.R. Fonollosa](https://www.kaggle.com/jarfo1), who let us use his own code as he was one of our former professors during the university.

#### **Model description**

Once we know which model we are going to use, let us give you a brew description of what an LSTM model is. It is quite important for you to understand the main characteristics of an LSTM model in order to have a better understanding of the entire idea of the model. Therefore, LSTM models are artificial neural networks which are used in AI and deep learning that were proposed by Hochreiter and Schmidhuber in 1997 as a solution to the vanishing gradients problem. One of the most outstanding features of these kind of models is that LSTM has feedback connections and not only can we process single data points with them, but also entire sequences of data (such speech data or videos). Moreover, LSTM architecture makes it easier for the Recurrent Neural Networks to preserve information over many time-steps.

#### Intended uses & Limitations

The most notable applications we can relate to LSTM are speech recognition, machine translation, connected handwriting recognition, parsing, image captioning, among others. In fact, when this model was first trained, his intended use was to classify the language of each sentence found in a Wikipedia dataset, mentioned in the Training Data section.

However, we can observe some limitations as LSTM models do not guarantee that there is no vanishing or exploding gradient, but it does provide an easier way for the model to learn long-distance dependencies. Moreover, other drawbacks of the LSTM approach can be found. For example, LSTM's take longer to train, require more memory to train too, are easy to overfit, the dropout is much harder to implement with these kinds of models and, they are sensitive to different random weight initializations. For this reason, while it is true that from 2013 to 2015, LSTM models became the dominant approach, in 2019 other approaches, such as Transformers, evolve into the dominant ones for certain tasks.

#### How to use

Since both our dataset and our model were obtained from the Kaggle repository, we will implement the same model using the same Kaggle repository and using the Python language. Therefore, as it could be expected when observing that we will be using Python, in order to execute the model, it is necessary for you to assure that you have the following libraries installed:

```{python}
import random
import numpy as np                      # linear algebra
import pandas as pd                     # data processing, CSV file I/O (e.g. pd.read_csv)
import torch                            # Deep learning framework
import torch.nn.functional as F
import time
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

Moreover, since the generation of the model relies on some randomness, we set a seed for reproducibility:

```{python}
#Init random seed to get reproducible results
seed = 1111
random.seed(seed)
np.random.RandomState(seed)
torch.manual_seed(seed)
```

Finally, to read any dataset when trying to implement our model, you can be guided by the following commands (you will notice that we will be working with two training subsets; x and y):

```{python}
INPUTDIR = 'the Kaggle directory where you have your datset stored'
print(os.listdir(f'{INPUTDIR}'))

# Any results you write to the current directory are saved as output
x_train_full = open(f'{INPUTDIR}/x_train.txt').read().splitlines()
y_train_full = open(f'{INPUTDIR}/y_train.txt').read().splitlines()
print('Example:')
print('LANG =', y_train_full[0])
print('TEXT =', x_train_full[0])
```

#### Limitations & Bias

Even if the training data used for this model could be characterized as fairly neutral we can suppose that LSTM models can not overcome biased datasets and these biases will also affect all fine-tuned versions of our model.

#### Training data

Actually, when describing a model, it is of vital importance to know which dataset we are going to use in order to know the goal of it and the reason why we are using this model and not another one. Our decision was to implement the model using the [Amazon Reviews dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews), that we have also extracted from the Kaggle repository. Therefore, the aim of our model will be to classify these reviews obtained from Amazon by taking review score 1 and 2 as a negative review, and 4 and 5 as a positive one (samples of score equal to 3 will be ignored). In the dataset, the negative reviews will be Class 1 and the positive ones will belong to Class 2. It is important to announce that we will be working with a very balanced dataset, as we have an equal amount of rows that are classified as both positive and negative reviews.

Nevertheless, our initial model was not trained with the dataset we will be using. José A.R. Fonollosa used a database created from [Wikipedia paragraphs](https://www.kaggle.com/competitions/wili5/data) when training his own model. This dataset, again, can be found in the Kaggle repository and can be split in three main files; *x_train.txt* and *x_test.txt*, where we can find 175.000 lines of text, where each line belongs to one language, and the *y_train.*txt where each line denotes the language of the same line.

#### Training procedure

##### Pre-processing

Before applying our model with our chosen dataset, we had to preprocess our data as we had more information than the one needed. In fact, the raw dataset had a total of 1.800.000 training samples for each class. Therefore, as we have 2 classes (the positive and the negative one), we were dealing with a training dataset of 3.6 million samples.

In order to reduce the size of this first version of our Amazon Reviews dataset, we are going to generate two more versions for different sizes of the dataset; the first one is going to be 1/3 of the whole dataset (1.2 million training samples) and the second one will be 1/4 of the entire dataset too (900.000 training samples).

Once we have checked the size of our dataset in terms of rows, it is time to start analyzing the size in terms of columns. In fact, our initial dataset has three columns: Polarity (class 1 for negative reviews or class 2 for the positive ones), Title (review heading) and Text (review body). Hence, for our purpose, we have just considered to use two of the three columns that we have mentioned before. Therefore, our y-class is going to be the Polarity and the x-class will be the Text (as we can observe, the Title columns is not going to be used). Moreover, we have also changed its names. So, from now on, Polarity column will be named as *class*, and the Text column will be named as *review_text*.

Finally, the last step we did during our preprocessing procedure was to define the type of our variables. The reason why we decided to apply this changes was due to the type the *class* column had. Initially, *class* was defined as a *int64,* while the values that were stored in it were just a 1 or a 2. Consequently, we have decided to modify this type and, from now on, the *class* column will be a *int8* variable. For the other column variable, we have just decided to keep the same type; *string.*

##### Pre-training

As it has been told before, our model was first trained by his own author [José A.R. Fonollosa](https://www.kaggle.com/jarfo1). When he did so, he used the following value parameters and the whole training lasts more or less 4h approximately:

```{python}
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 64
BATCH_SIZE = 256
TOKEN_SIZE = 200000
EPOCHS = 25
MAX_NORM = 1
```

#### Evaluation results

The model achieves the following results without any fine-tuning:

-   TRAINING ACCURACY OF THE MODEL IS: **96.372%**

-   TRAINING TIME: **4h 1 min**

Once we have fine-tuned our model with MLflow, the training accuracy we have obtained is:

-   TRAINING ACCURACY OF THE MODEL IS:

-   TRAINING TIME:
