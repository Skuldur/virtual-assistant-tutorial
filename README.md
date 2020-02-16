# Basic Virtual Assistant

Basic Virtual Assistant is the accompanying Github repository for my blog series "Making Your Own Alexa". The code in this repository can be used as a basis for a virtual assistant project.

Part 1: <LINK TO PART 1>

## Getting Started

```
pip install -r requirements.txt
```

## Usage

The intent_trainer module contains a CLI for training new intents. It has a few conditional options that can be used:

* *--schema_file* is the path to the training data you want to use. E.g. `commands/play_commands.json`.
* *--name* is the name of the intent that you're training.
* *--sample_size* is the amount of rows that should be saved so that they can used to train the intent classification model.
* *--batch_size* is the size of the mini-batches.

Below is an example on how you can train an Named Entity Recognition model for the Play command.

```
python intent_trainer.py --schema_file commands/play_commands.json --name play --sample_size 400 --batch_size 128
```



