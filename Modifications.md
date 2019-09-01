# Modifications and Notes for STC

Christian Ritter

* BERT by Devlin19, implementation of https://github.com/google-research/bert

## Comments

* Implementation is done through the TF Estimator API and a custom Estimator is being built 
(see https://www.tensorflow.org/guide/custom_estimators)
*  It is currently not possible to re-produce most of the BERT-Large results on the paper 
using a GPU with 12GB - 16GB of RAM
* For fine tuning a GPU like GTX 1080 can be used, easy with our infrastructure 
* The released models were trained with sequence lengths up to 512,
* * Signatures: Tensorflow ONLY run those parts of the module that end up as dependencies of a target in tf.Session.run()
* Modified binary prediction with predicting_movie_reviews_with_ber_hub.py to work with offline tokenizer and model
* Added multi-label-classification-bert.ipynb from another repo, for potential application in multi-class cases
* I commented lines 460 in run_classifier.py as this debug information made development difficult


## Model Data

* 2 options: from TFHub or by downloading (indirectly from TFHub)

* Direct download to get zip file ()
    * included are 3 important files as follows:
        * BERT_VOCAB= ‘uncased-l12-h768-a12/vocab.txt'
            * contains the vocabulary
        * BERT_INIT_CHKPNT = ‘uncased-l12-h768-a12/bert_model.ckpt’
            * standard TF checkpoint
        * BERT_CONFIG = ‘uncased-l12-h768-a12/bert_config.json’
            * contains configuration parameters incl. hyperparameters such as hidden_size
 

* Use TFHub data (either through python or by downloading)
    * downloaded version in : 

### Variations of the model

* download link at https://github.com/google-research/bert
* uncased_L-12_H-768_A-12:
    * lowercase
    * Specs: 1 language, 12-layer, 768-hidden, 12-heads, 110M parameters, 30500 vocabulary
    * 423MB
* multi_cased_L-12_H-768_A-12 This is a multi-lingual (incl. French) version 
    * the cased version is the recommeded one (see github/bert comments)
    * Specs: 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters, 119547 vocabulary
    * 680Mb
* BERT-Large, Uncased (Whole Word Masking):
    * with new technique: whole word masking
    * much larger in comparison with other models
    * Sepcs: 24-layer, 1024-hidden, 16-heads, 340M parameters 
    * 1.2GB
    * 
    
    
## Input Data


* acllmdb: movie review dataset, http://ai.stanford.edu/~amaas/data/sentiment/



## What needs to be done



## Modification for Multi-label classification:

* https://towardsdatascience.com/building-a-multi-label-text-classifier-using-bert-and-tensorflow-f188e0ecdc5d
* https://github.com/javaidnabi31/Multi-Label-Text-classification-Using-BERT/blob/master/multi-label-classification-bert.ipynb