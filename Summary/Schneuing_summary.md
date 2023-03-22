# STRUCTURE-BASED DRUG DESIGN WITH EQUIVARIANT DIFFUSION MODELS
- generates novel ligands conditioned on protein pockets
- 3-D condditional diffusion model - respects translation, reflection, rotation and premutation
- 2 strategies:
    - protein-conditioned generation
        - protein as fixed context
    - ligand-inpainting generation

- atomic point clouds
    - coordinates
    - categorical features (type of atom etc.)

- 3 dimensional context in each step of denoising process 
- parametrize the noise by predictor with EGNN
- process ligand and pocket nodes with single GNN, atom types and residue types are first embedded in a joint
node embedding space by separate MLP
- need to make sure that protein context is fixed
- pocket centered at origin
- doing this for all training data , translation equivariance becomes irrelevant
- for sampling and evaluation of inference, we can just move molecules
