# Basic Virtual Assistant

Basic Virtual Assistant is the accompanying Github repository for my blog series "Making Your Own Alexa". The code in this repository can be used as a basis for a virtual assistant project.

Part 1: https://towardsdatascience.com/making-your-own-alexa-entity-extraction-8c7f23eb65a

## Getting Started

```
pip install -r requirements.txt
```

NLTK will require you to download a data for its tokenizer so open Python in your terminal and run the following code

```
import nltk
nltk.download('punkt')
```

## Usage

The intent_trainer module contains a CLI for training new intents. It has a few conditional options that can be used:

* `task` is the task you want to run (`train` or `predict`).
* `--schema_file` is the path to the training data you want to use. E.g. `commands/play_commands.json`.
* `--name` is the name of the intent that you're training.
* `--sample_size` is the amount of rows that should be saved so that they can used to train the intent classification model.
* `--batch_size` is the size of the mini-batches.
* `--epochs` is the number of epochs the model should train for.
* `--command` is the command you want to predict the labels for.

Below is an example on how you can train an Named Entity Recognition model for the Play command.

```
python intent_trainer.py train --schema_file commands/play_commands.json --name play --sample_size 400 --batch_size 128 --epochs 15
```

And here is an example on how to test one of your commands.

```
python intent_trainer.py predict --name play --command "play let it be by the beatles"
```



