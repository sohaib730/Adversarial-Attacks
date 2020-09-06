# Adversarial-Attacks
## Adversarial Attacks:
Implemented for CIFAR10, but can easily be modified for any other dataset.

### Custom attacks:

-	Model: CIFAR CNN model defined using keras  (model should be trained before generating attacks)
-	Two white box attacks based on technique CW0 and CW2
- Two Black box attacks Momentum based attacks and PGD 
- To run code type: python Custom_Attacks/Generate_Adversarial.py 

### Generated using Cleverhans Library:

- Model: Wide ResNet using Tensorflow (Model should be trained before generating adversarial examples)
- Use cleverhans library to generate deepFool, PGD and FGSM based attacks.
- Important feature is batch implementation deepFool attacks, which is not straight forward like all other attacks
- To run Code type: python Cleverhans_Attacks/GenAdv_usingCleverhans.py
