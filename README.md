# Hyperface Gender Classification
Based on the work of Ranjan, Patel and Chellappa (2016):  
[HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition](https://github.com/aleju/papers/blob/master/neural-nets/HyperFace.md)
  
Best result obtained: 93% validation accuracy 
## Training
In trian/train.py change path to desired location. Input data need to be square, the preprocessing functions only scale not crop.
  Then run `python train.py`
  
## Testing
In test/hyperface_test.py change the path to test data then run `python hyperface_test.py`.
