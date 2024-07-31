# CS7643_Project_Repo

List of directory / contributions

```txt
|- fixmatch-pytorch        # Folder containing Jintae Park's Input. Forked from FixMatch-pytorch repository.
|    |- train.py            # Contains main logic and modification by Jintae Park. 
|                             Use with additional arguments for freematch implementation: 
|                             --freematch, --ema, --disable-saf, --lambda-f
|
|- fix_match_AB        # Folder containing Apratim Bajpai's implementation of FixMatch.
|    |- main.py            # Function required to run FixMatch on Cifar10 dataset.
|    |- datasets.py            # Function required to generate labelled and unlabbeled Cifar10 dataset.
|    |- ema.py            # Implementation of Exponential moving averages
|    |- get_wide_resnet.py            # Creating a wide resnet
|    |- randaugment.py            # Implementation of random augmentation
|    |- test_train.py            # Function for training and evaluation
|    |- utils.py            # Collection of utility functions
|
|- extension_ideas          # Folder containing subfolders for Augmentation Anchoring & Distribution Alignment experiments
|    |- augmentation_anchoring/     # Folder containing Yujeong Lozalee & Xander Kehoe's input. Forked from FixMatch repository.
|    |    |- fixmatch.py             # Contains main code modifications for incorporating Augmentation Anchoring into FixMatch. 
|    |                               		(fixmatch_aa_ver1.py & fixmatch_aa_ver2.py files are archived code files)
|    |    |- libml/
|    |    |    |- augment.py         # Contains minor code modifications
|    |    |    |- utils.py           # Contains minor code modifications
|    |- distribution_alignment/      # Folder containing Yujeong Lozalee's input. Forked from FixMatch repository.
|    |    |- fixmatch.py             # Contains main code modifications for incorporating Distribution Alignment into FixMatch. 
|
|- (Folder for Xander)
```
