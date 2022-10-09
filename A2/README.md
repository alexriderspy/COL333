# Interaction with the Game Simulator

The following command will initiate a game of ExtendedConnect4 between **agent_1** and **agent_2** with the initial state given at the **test_case_path**. The script can be invoked with several options described subsequently. The command should display the GUI showing the game. This time available to make a move is controlled by parameter `time`. 

```python
python3 -m connect4.ConnectFour {agent_1} {agent_2} {test_case_path} --time {time_in_seconds_per_step}
```

The following command will allow a **human agent** to play against the **AI agent.** The human agent (you) can type in a move as *"1"* or *"1p"*, where “1” indicates the column number. The presence or absence of “p” after the column number indicates if the move is a *PopOut* or a regular move. 

```python
python3 -m connect4.ConnectFour ai human connect4/initial_states/case4.txt --time 20
```

A simple **random agent** is provided in the starter code. The random agent simply picks its moves uniformly at random among the available ones. A game between an AI agent and the random agent can be initiated as follows. 

```python
python3 -m connect4.ConnectFour ai random connect4/initial_states/case4.txt --time 20
```

You might also see your bot play against itself.

```python
python3 -m connect4.ConnectFour ai ai connect4/initial_states/case4.txt --time 20
```

Moves that violate constraints are considered **invalid moves**. If a player attempts to play an invalid move, the game simulator does not change the game state (i.e., the attempted move is skipped) and the turn switches to the next player. Examples of invalid moves: (i) using more than the permitted **K** PopOut moves (ii) performing a PopOut move in an invalid column (Player1 can perform PopOut only in the even columns and Player2 only in the odd columns) and (iii) exceeding the permitted time duration per move (parameter T) in deciding your action.