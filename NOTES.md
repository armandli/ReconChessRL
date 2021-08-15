### Problems
1. encoding 1 does not allow shared experience between black and white player. this is because black and white actions are different. we could see if we can do orientation transformation to make the actions the same but act differently if it's black or white in order to allow shared experience; this means we'll have to change the orientation before feeding it into the action model and change the orientation back after taking the action produced by the action model; orientation does not affect sense, where the square is the same regardless of if it's white or black player, but the sense action should definitely different depending on whether the player is black or white
2. encoding should let the agent know if it's going first or second, i.e. black and white should have one bit of encoding difference
3. 4096 actions has a lot of permanently dead actions due to unreachable move no matter what piece it is, we can reduce it down to 1700ish but requires a lot more mappping code; should speed up training a bit
4. use embedding for action space ? so that any action output from the model can be mapped dynamically to a valid action, so there is no sample inefficiency from invaid actions
5. internal state representation for the recon chess observation just like MuZero. modify MuZero to adapt to Reconchess
6. add underpromotion to move set? for only knight underpromotion?

#### Using MuZero for Reconchess
1. function h needs to be able to take event sequences and update the internal state representation, the events include oppo capture, self capture, self move, sense
   update, for oppo capture, there need to be a state where oppo moved but not captured anything
2. the internal state representation now means a set of states, instead of just one possible theoretical state of the game
3. because during state transition, the set of possible states before and after a action is taken would change, this means function g will need to be able to allow
   stochasticity and is not a determinsitic state transition function, where mutliple next state is possible from one action transition representing differences in
   the set of real game states has changed in each internal state representation output
