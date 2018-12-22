
# PacmanDQN & GhostDQN
Deep Reinforcement Learning for training Non - Player Characters i.e. Ghosts in Pac-man

## Demo
### Independent
#### Single ghost against a trained Pacman Agent
![Demo](https://github.com/adityachamp/DL_RL_CollisionAvoidance/blob/master/PacmanDQN/videos/independent_single_ghost.gif)
#### Two ghosts
![Demo](https://github.com/adityachamp/DL_RL_CollisionAvoidance/blob/master/PacmanDQN/videos/independent_two_ghosts.gif)

### Teamwork
#### Two ghosts against a trained Pacman agent
![Demo](https://github.com/adityachamp/DL_RL_CollisionAvoidance/blob/master/PacmanDQN/videos/teamwork_two_ghosts.gif)
#### Three ghosts against a trained Pacman agent
![Demo](https://github.com/adityachamp/DL_RL_CollisionAvoidance/blob/master/PacmanDQN/videos/teamwork_three_ghosts.gif)
## Example usage

Run a model on `mediumGrid` layout for 6000 episodes, of which 5000 episodes
are used for training.

```
$ python3 ghosts.py -p PacmanDQN -g ghostDQN -n 6000 -x 5000 -l mediumGrid

```

### Layouts
Different layouts can be found and created in the `layouts` directory.
This repository utilises the smallGrid, mediumGrid and mediumClassic. 

### Parameters

Parameters can be found in the `params` dictionary in `ghostDQN_Agents.py`. <br />
 <br />
Models are saved as "checkpoint" files in the `/saves` directory. <br />
Load and save filenames can be set using the `load_file` and `save_file` parameters. <br />
After changing the layout, change the value of the height and width in ghostDQN_Agents.py file. <br />
For multiple ghosts, the checkpoints are saved as `model-ghost(ghost_index)medium_classic_3_ghosts`
 <br />
Episodes before training starts: `train_start` <br />
Size of replay memory batch size: `batch_size` <br />
Amount of experience tuples in replay memory: `mem_size` <br />
Discount rate (gamma value): `discount` <br />
Learning rate: `lr` <br />
 <br />
Exploration/Exploitation (ε-greedy): <br />
Epsilon start value: `eps` <br />
Epsilon final value: `eps_final` <br />
Number of steps between start and final epsilon value (linear): `eps_step` <br />

## Citation

Please cite this repository if it was useful for your research:

```
@article{van2016deep,
  title={Deep Reinforcement Learning in Pac-man},
  subtitle={Bachelor Thesis},
  author={van der Ouderaa, Tycho},
  year={2016},
  school={University of Amsterdam},
  type={Bachelor Thesis},
  pdf={https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf},
}

```

* [van der Ouderaa, Tycho (2016). Deep Reinforcement Learning in Pac-man.](https://esc.fnwi.uva.nl/thesis/centraal/files/f323981448.pdf)

## Requirements

- `python==3.5.1`
- `tensorflow==0.8rc`

## Acknowledgements

DQN Framework by  (made for ATARI / Arcade Learning Environment)
* [deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/mrkulk/deepQN_tensorflow](https://github.com/mrkulk/deepQN_tensorflow))
* [PacmanDQN](https://github.com/mrkulk/deepQN_tensorflow) ([https://github.com/tychovdo/PacmanDQN/](https://github.com/tychovdo/PacmanDQN/))

Pac-man implementation by UC Berkeley:
* [The Pac-man Projects - UC Berkeley](http://ai.berkeley.edu/project_overview.html) ([http://ai.berkeley.edu/project_overview.html](http://ai.berkeley.edu/project_overview.html))
