# Progress Log

### Entry 1/10/2022
Results from last semester indicate that L2 loss is not suitable for the midi VAE task. We wish to design a new loss function that better encapsulates midi structure and how close the representations are to the original tensor. 

We propose a dynamic programming algorithm, similar to the Wagner-Fischer algorithm for computing the edit distance between strings. We must account for the following operations: 

1. **Deletion:** deleting one or multiple notes
2. **Insertion:** inserting one or multiple notes
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