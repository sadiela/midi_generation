# Progress Log

### Entry 1/10/2022
Results from last semester indicate that L2 loss is not suitable for the midi VAE task. We wish to design a new loss function that better encapsulates midi structure and how close the representations are to the original tensor. 

We propose a dynamic programming algorithm, similar to the Wagner-Fischer algorithm for computing the edit distance between strings. We must account for the following operations: 

1. **Deletion:** deleting one or multiple notes at a given time step
2. **Insertion:** inserting one or multiple notes at a given time step 
3. **Substitution:** substituting one or more notes for one or more other notes
4. **Pitch shift:** shifting all notes up or down by a certain number of pitches

Our algorithm must account for several extra factors that the WF algorithm does not: 
* **Pitch Relationships:** In the case of strings, distance is binary. If two characters are the same, the distance is 0, and if they are different, the distance is one. In our case, pitches can be different distances apart. Ideally, our loss function can take this into account (maybe even in an intelligent way)
* **Pitch Shifting:** We introduce a new operation, pitch shifting, where we move all of the pitches up by the same amount. We need a way to incorporate this new operation into our algorithm.
* **Multiple Notes**: There can be two "characters" (pitches) cooccuring in the same time slot. 
* **No Notes**: There can be no notes playing at any given time. We can account for this possibly by counting it as its own note
* **Note Lengths**: It is also desirable to account for the notes being the correct length.


Eventually, we also need to find some kind of differentiable approximation to our loss function so we can backpropagate through it. 


**Open Questions**:
1. How can we make our algorithm differentiable? 
2. How can we make sure note position is prioritized over note length? 
3. How does this problem reduce to finding the highest-scoring path between the start and end node on a DAG? 
4. How can we test our "loss function"?


### Entry 1/11/2022

- Shift somewhat expensive, but cheaper than shifting individual notes or adding/subtracting
- Work with synthetic MIDI tensors for testing 
- New evaluation function? Use DP loss as evaluation if confident
- Code for simpler DP problem for understanding

### Entry 1/18/2022

To Do: 
* Set up logging (instead of print statements) to go with verbosity command line flag (DONE)
* Get two recursive implementations of knapsack DP to match answers --> DONE(?)
* High level explanation of VQ-VAE algorithm
* Deep dive into algorithm architecture
    * Model diagram
* Outline parts of the architecture that can be experimented with and what needs to stay the same
* Successful run of VAE (no quantization)
* Documentation/organization of all scripts (MOSTLY DONE)
    * vq_vae deep dive still left
* Add Yichen to textconv project (DONE)
* First tasks for Yichen (DONE)
* Decide how/where we want Yichen to work on the SCC... clone another copy of the repo? (DONE)
* Create a baby dataset for testing (3 songs?)
* Create a medium-sized dataset for Yichen (50 songs?)
* Create clean data preprocessing pipeline (DONE)
    * Add more command line arguments

### Entry 1/22/2022
Finished differentiable recursion for the edit distance problem, including code for generating the theta matrix from two words. Next step is to adapt this implementation for the midi problem. The structure will be the same, but we have to account for the pitch shift operation as well. We also need to change the value of the edges to account for how much of a change is required to add/subtract/change a note (same number of notes before and after? How much do they have to be moved?).
