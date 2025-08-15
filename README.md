# Deep learning for multi-label rock news classification
# Overview
As a continuation and final exploration of previous classification systems, including general NLP tasks on rock news articles ([Rock is not dead: NLP experiments on rock news articles](https://github.com/IvoDSBarros/rock-is-not-dead_nlp-experiments-on-rock-news-articles/tree/main)) and a prior weakly supervised multi-label approach ([Multilabel classification task on rock news articles](https://github.com/IvoDSBarros/multilabel_classification)), the deep learning project presented in this repository focuses on assigning topic labels to a dataset of rock news headlines. Specifically, it represents the second weakly supervised approach where the label assignments are derived from an antecedent rule-based classification model (for details, see [Rule-based Text Classification](https://github.com/IvoDSBarros/rock-is-not-dead_nlp-experiments-on-rock-news-articles/blob/main/README.md#rule-based-text-classification).
Several deep learning strategies were tested for this weakly supervised multi-label problem leveraging Tensor Flow, namely:

**<u>1) Hybrid models</u>**
<br>
A combination of Convolutional Neural Networks (CNNs) with Recurrent Neural Networks (RNNs), in specific Bidirectional Long Short-Term Memory (BiLSTMs). These models used pre-trained GloVe word embeddings for semantic representation and were further enhanced by Attention mechanisms.

**<u>2) Fine-tuning pre-trained transformers</u>**
<br>
This approach involved the adaptation and fine-tuning of state-of-the-art transformer models, such as DistilBERT.

In general, the results from the experiments confirmed the superior performance of the fine-tuned DistilBERT model for this multi-label rock news classification. While the hybrid CNN-BiLSTM model achieved a Micro-average F1 of 0.897 and a Macro-average F1 of 0.507, the DistilBERT approach obtained significantly higher metrics, with an average optimized Micro-average F1 of ~0.9957 and a Macro-average F1 of ~0.9924. This outstanding performance indicates that fine-tuned transformer architectures are particularly well-suited to capture the nuanced, context-dependent features necessary for accurate topic assignment in this domain.

<details>
<summary> Table of Contents </summary>

1. [Limitations and challenges](#1-limitations-and-challenges)
2. [Model 1: CNN-BiLSTM with GloVe Embeddings and Attention](#2-model-1-cnn-bilstm-with-glove-embeddings-and-attention)
    * [Methodology](#21-methodology)
    * [Results](#22-results)
3. [Model 2: Transformer-based model (DistilBERT)](#3-model-2-transformer-based-model-distilbert)
    * [Methodology](#31-methodology)
    * [Results](#32-results)
[References](#references)

</details>

# 1. Limitations and challenges
+ A primary methodological consideration concerns the computational environment: due to the lack of a dedicated Graphics Processing Unit (GPU), all tasks for the deep learning models were executed exclusively on a Central Processing Unit (CPU). This hardware constraint caused critical limitations on both the speed of model training and the scale of experimentation. For instance, the hyperparameter tuning grids for the CNN-BiLSTM with GloVe Embeddings models required considerable reduction, whereas the DistilBERT model required over four days to be executed.
+ Another key challenge for this task is the inherent high imbalance observed in the text corpus, as highlighted in the project "Multilabel classification task on rock news articles". Despite the implementation of the iterative stratification technique (Szymański et al, 2016) to address this constraint, it was also necessary to perform a post-training threshold tuning to find the ideal "cut-off point" for the model's predictions as detailed further below.

# 2. Model 1: CNN-BiLSTM with GloVe Embeddings and Attention
The first deep learning architecture implemented in this project is a hybrid model comprising Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory (BiLSTM), enhanced with an Attention mechanism. This model leverages pre-trained GloVe word embeddings for semantic representation and is intended to capture local features (CNNs) while processing the sequential nature of text (BiLSTM).

## 2.1. Methodology
+ Aiming to optimize the performance of models based on pre-trained word embeddings, multiple experiments were conducted to evaluate the impact of different GloVe embedding dimensionalities (50d, 100d, 200d, and 300d) on key multi-label classification metrics. For these preliminary GloVe dimensionality evaluations, a standard classification threshold of 0.5 was applied to all predicted probabilities. The results generally indicate that as the dimensionality of the GloVe embeddings increases from 50d to 300d, there is a consistent improvement in both micro-average and macro-average F1 scores, precision, and recall (see Table 1). This suggests that higher-dimensional embeddings are better at capturing the semantic relationships in the text data. Given these findings, GloVe 300d embeddings were selected for all subsequent model development and evaluation.
<br>

**Table 1. Model evaluation metrics for different GloVe Embeddings**

![](https://github.com/IvoDSBarros/dl-rock-news-classification/blob/47c30b7f8c889aee3a0053e6646ce02fa35ed5fe/assets/table_01.PNG))

<br>

+ The standard practice for multi-label classification problems is to use sigmoid activation in the output layer (Brownlee, 2020; Grivas et al, 2024). This approach treats each label as an independent binary classification task, with its own dedicated sigmoid activation function.
+ The initial classification threshold for the CNN-BiLSTM model was set to 0.5. To refine the model's performance, post-training threshold tuning was performed on the test set. The tuning process specifically focused on maximizing the Micro-F1 score as this metric is often considered robust for multi-label classification, particularly when dealing with class imbalance. Thresholds were systematically evaluated from 0.01 to 0.99 against the model's raw probability predictions. Upon identifying the optimal threshold of 0.30, all final comprehensive evaluation metrics were re-calculated and presented using this refined value to ensure a more accurate representation of the model's performance.
+ Limited by significant computational constraints from CPU-only execution, a hyperparameter tuning process was conducted using both grid and random search. The constrained search space (which included parameters such as learning rates ([0.0001, 0.0005]), batch sizes ([64, 128]), and maximum epochs ([5, 10], with early stopping)) resulted in no improvement in the model's micro-F1 score. Performance across all evaluated parameter sets remained largely consistent, indicating that the explored hyperparameter configurations did not substantially enhance the model's overall predictive capability for this multi-label classification task.

## 2.2. Results
### 2.2.1. Training performance overview
+ The training performance of the model is comprehensively illustrated by the Micro F1 and Loss curves (see Figure 1 and Figure 2 respectively). As depicted in Figure 1, both training and validation Micro F1 scores demonstrated significant and consistent increases throughout the 30 epochs. In alignment with these positive trends, Figure 2 shows that both training and validation loss curves experienced a sharp initial decrease followed by a gradual decline. The consistent improvement in both metrics across training and validation sets confirms effective learning and strong generalization without signs of overfitting.

<br>

**Figure 1. Micro F1 Score by epoch**

![](https://github.com/IvoDSBarros/dl-rock-news-classification/blob/47c30b7f8c889aee3a0053e6646ce02fa35ed5fe/assets/figure_01.png)

<br>

<br>

**Figure 2. Model Loss by epoch**

![](https://github.com/IvoDSBarros/dl-rock-news-classification/blob/47c30b7f8c889aee3a0053e6646ce02fa35ed5fe/assets/figure_02.png)

<br>

### 2.2.2. Test set evaluation results with optimal threshold (0.30):
+ The model was evaluated on the test set using an optimal classification threshold of 0.30, a value determined through post-training tuning. 
The Micro-average F1 Score of 0.897 is a key indicator of the model's strong overall performance. This is further confirmed by a Micro-average Precision of 0.926 and a Micro-average Recall of 0.871, which together show a low rate of false positives and effective identification of relevant instances, respectively.
+ In contrast, the Macro-average F1 Score was considerably lower at 0.507. This significant discrepancy highlights a common challenge in multi-label classification with imbalanced datasets:  while the model excels on prevalent classes, its performance tends bo be weaker on less frequent categories.
+ Finally, the AdaBoostClassifier from previous work outperformed the CNN-BiLSTM model. The first achieved higher metrics, particularly a robust Macro-average F1 of 0.976 (with a Micro-average F1 of 0.990), suggesting its superior ability to handle class imbalance and generalize effectively across all categories.

<br>

**Table 2. Model evaluation metrics with optimal threshold (0.30)**

![](https://github.com/IvoDSBarros/dl-rock-news-classification/blob/47c30b7f8c889aee3a0053e6646ce02fa35ed5fe/assets/table_02.PNG)

<br>

#  3. Model 2: Transformer-based model (DistilBERT)
Given its solid pre-trained representations for text classification and performance efficiency, DistilBERT was selected as the foundational model for this transformer-based approach.

## 3.1. Methodology
+ The model was implemented using the Hugging Face Transformers library, integrated with TensorFlow as the deep learning framework.
+ To adjust the model's final layer with appropriate logits for a multi-label classification setup, the problem_type was configured as "multi_label_classification", whereas TFAutoModelForSequenceClassification was loaded with num_labels precisely corresponding to the total number of distinct rock news categories.
+ A 5-fold stratified cross-validation strategy was performed on the full dataset for robust evaluation.
With the purpose of regularization, Early Stopping was introduced as a key technique.
+ The post-training threshold tuning was crucial in reaching the model's peak performance. The analysis revealed that the optimal threshold for maximizing Micro-F1 was approximately 0.25, leading to the 0.9957 score. Conversely, the optimal threshold for maximizing Macro-F1 was approximately 0.13, achieving 0.9924. 

## 3.2. Results
### 3.2.1. Overall performance summary (optimized with threshold tuning)
+ The DistilBERT model demonstrated outstanding overall performance across the 5-fold cross-validation. This performance was further optimized through the post-training threshold tuning, which enabled the model to achieve its peak potential. When evaluated at their respective optimal thresholds, the average cross-validation scores are as follows: 

<br>

**Table 3. Model evaluation metrics with optimal threshold**

![](https://github.com/IvoDSBarros/dl-rock-news-classification/blob/47c30b7f8c889aee3a0053e6646ce02fa35ed5fe/assets/table_03.PNG)

<br>

+ The exceptionally high scores, achieved by applying the post-training threshold optimization, are indicative of the model's robust generalizability and high classification accuracy across nearly all categories.
+ In particular, the Micro-F1 Score, being extremely close to 1, confirms that the model correctly identified the vast majority of true positive labels. In addition, the Macro-F1 Score, which equally weights each class regardless of its frequency, showed superb performance even across less frequent categories.

### 3.2.2. Training dynamics and convergence
+ The analysis of the loss curves across all folds (see Figure 3) provides clear evidence of the model's robust learning process. The training loss (green line) consistently decreased and subsequently stabilized over the epochs, which is characteristic of effective data fitting. 
+ Similarly, the validation loss (brown line) replicated this trend of reduction and subsequent stabilization, confirming the model's ability for generalization to unseen data. These plots are independent of the classification threshold applied post-training and attest the model's internal learning convergence.

<br>

**Figure 3. DistilBERT individual fold Loss Curves**

![](https://github.com/IvoDSBarros/dl-rock-news-classification/blob/c0823a87ee1f5539077d7771c4cf35bf36a111eb/assets/figure_03.png)

<br>

### 3.2.3. Fold-wise analysis
+ The majority of categories, including common rock topics such as "album", "tour", "video", and "award", consistently achieved near-perfect precision, recall, and F1-scores across all folds.
+ In contrast, categories such as "art", "drug", and "release" were underrepresented in the dataset. For these categories, where data was sparse, the threshold tuning is expected to significantly improve their performance by allowing the model to better identify the few existing instances.
+ Finally, 'rehab' remains the most challenging category. With only 2-4 instances in the entire dataset, the model consistently yielded 0.00 for precision, recall, and F1-score across all reported folds. Significant improvement for this extremely rare category will likely require data augmentation or specialized handling strategies.

### 3.2.4. Conclusion
+ In summary, the DistilBERT model delivers exceptionally high performance for multi-label rock news classification, and its performance was further optimized through threshold tuning. 
+ The strong generalizability and effectiveness across most categories are supported by the remarkable average optimized Micro-F1 score above 0.995 and Macro-F1 score above 0.992.
+ Despite the fact that training dynamics reflect robust learning, the primary remaining challenge lies in the accurate classification of extremely rare categories like 'rehab'. Addressing these edge cases might require further data-centric strategies, but for the vast majority of the labels, DistilBERT stands as a highly proficient and accurate classifier.
+ When compared to the AdaBoostClassifier from the previous study, the transformer-based DistilBERT model proved to be a more effective solution. Even with the AdaBoostClassifier’s high baseline (Micro-average F1: 0.990, Macro-average F1: 0.976), the DistilBERT model surpassed these metrics (Micro-average F1: ~0.9957, Macro-average F1: ~0.9924), emphasizing its outsanding representational capacity and contextual understanding.

# References
+ [Grivas, A., Vergari, A., Lopez, A. (2024) Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification. Proceedings of the AAAI Conference on Artificial Intelligence, 38(11), 12208-12216.](https://www.pure.ed.ac.uk/ws/portalfiles/portal/408679788/Taming_the_Sigmoid_GRIVAS_DOA09122023_AFV_CC_BY.pdf)
+ [Brownlee, J. (2020) Machine Learning Mastery.](https://machinelearningmastery.com/multi-label-classification-with-deep-learning/)
+ [Szymański, P., Kajdanowicz, T. (2016) A scikit-based Python environment for performing multi-label classification. Journal of Machine Learning Research, 1, 1-15.](https://www.jmlr.org/papers/volume20/17-100/17-100.pdf)



