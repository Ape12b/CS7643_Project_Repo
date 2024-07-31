# CS7643_Project_Repo

List of directory / contributions

```txt
|- fixmatch-pytorch        # Folder containing Jintae Park's Input. Forked from FixMatch-pytorch repository.
|    |- train.py            # Contains main logic and modification by Jintae Park. 
|                             Use with additional arguments for freematch implementation: 
|                             --freematch, --ema, --disable-saf, --lambda-f
|
|- (Folder for AB)
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
