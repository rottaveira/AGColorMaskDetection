# AGColorMaskDetection
AG - Color mask detection

This work was inspired on this paper: https://www.researchgate.net/publication/336725889_Coffee_Leaf_Rust_Detection_Using_Genetic_Algorithm
But, instead use kernel mask approach, i had used a color range approach.
The verification method was validated applying dice algorithm between masks.

The AG implementation got the parameters bellow:
100 generations
bests 60% childrens + 40% parents for each generation
0.06 mutation rate
Using roullete selection method
