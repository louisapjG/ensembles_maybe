# ml_toolbox
A mix of tools to test and compare:

  *Ensembles methods 
  
  *Filtering algorithm based on "Cognitive Diversity" to pass to Ensembles methods
  
  *Parameters fitting strategies (Gridsearch,...)
 

Board_opinion is class that will handle the training of a set of pipelines. It handles variabilities filtering, normalization, dimension reduction, and finally the classifiers themselves. It will generate every combinations of variabilities, normalization , dimension reduction and classifiers given. NOT every combination of parameters just the algos.


Ensemble method: 
  
  *New_mgs contains the class MGS for Mixed Group Scores handling a modified soft + beta implementation of the Mixed Group Ranks algo.
  
  
  *For now other ensembles method are not defined here.
  
  
Filtering methods:

  Performance: Select x, given number, of clfs based on performance. 
  CDS: Cognitive Diversity Strength as defined in "Preference Prediction Based on Eye Movement Using Multi-layer Combinatorial Fusion".
  Sliding Ruler: Ranks by performance a separately by CDS before selecting the X highest common to both.
