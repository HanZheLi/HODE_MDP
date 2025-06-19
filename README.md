# HODE-MDP
The repository for Hypergraph ODE-based Multi-aspect User Dynamic Preference Modeling for Next POI Recommendation.

## Requirements

+ Pytorch 1.13.0
+ torchdiffeq 0.2.3
+ scikit-learn 0.23.2
+ scipy 1.10.1
+ pandas 1.5.3
+ numpy 1.24.3

## Running 
run HODE-MDP on NYC: `python run.py --dataset NYC --t1 7 --t2 14 --t3 7 --lambda_cl 0.4`
run HODE-MDP on TKY: `python run.py --dataset TKY --t1 7 --t2 5 --t3 7 --lambda_cl 0.3`
