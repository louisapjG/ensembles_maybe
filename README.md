# ml_toolbox
A mix of tools to facilitate the setting up and testing of:

  *Ensembles methods 
  
  *Filtering algorithm based on "Cognitive Diversity" to pass to Ensembles methods
  
  *Parameters fitting strategies
 


Board_opinion is class that will handle the training of a set of pipelines. It handles variabilities filtering, normalization, dimension reduction, and finally the classifiers themselves. It will generate every combinations of variabilities, normalization , dimension reduction and classifiers given. NOT every combination of parameters just the algos.


Ensemble method: 
  
  *New_mgs contains the class MGS for Mixed Group Scores handling a modified soft + beta implementation of the Mixed Group Ranks algo.
  
  
  *For now other ensembles method are not defined here.
  
  
Filtering methods:

  Performance: Select x, given number, of clfs based on performance. 
  
  To be implemented: Cognitive Diversity Strength (as defined by Schweikert C., Gobin L., Xie S., Shimojo S., Frank Hsu D. (2018) Preference Prediction Based on Eye Movement Using Multi-layer Combinatorial Fusion. In: Wang S. et al. (eds) Brain Informatics. BI 2018. Lecture Notes in Computer Science, vol 11309. Springer, Cham)
