# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

## Overview
This project implements a simple content-based recommendation system that suggests movies based on a user's textual description of their preferences. The system uses TF-IDF vectorization and cosine similarity to match user input with movie descriptions from a dataset.

## Dataset
The dataset used in this project is a local CSV file named `imdb_top_1000.csv`, which contains information about the top 1000 movies from IMDB. The dataset includes columns such as `Series_Title`, `Overview`, `Genre`, `Director`, `Star1`, `Star2`, `Released_Year`, and `IMDB_Rating`.

If the local dataset fails to load, the system will fall back to a small sample dataset with two movies.

## Setup
### Python Version
This project requires Python 3.x.


## Running the Code
To run the recommendation system, execute the following command:

```bash
python recommendation.py
```

The system will prompt you to describe the kind of movie you're in the mood for. Type your description and press Enter. The system will return the top 5 movie recommendations based on your input.

To exit the program, type `exit` when prompted.

## Example Output
Here is an example of how to interact with the system and the expected output:

```
Welcome to the Enhanced Movie Recommender!

Describe what kind of movie you're in the mood for (type 'exit' to quit):
> I love thrilling action movies set in space, with a comedic twist.

Top Recommendations Based on Your Preferences:
--------------------------------------------------

1. Shin seiki Evangelion Gekijô-ban: Air/Magokoro wo, kimi ni (1997)
   Genre: Animation, Action, Drama
   IMDB Rating: 8.1
   Match Score: 0.09
   Synopsis: Concurrent theatrical ending of the TV series Shin seiki evangerion (1995)....

2. The Incredibles (2004)
   Genre: Animation, Action, Adventure
   IMDB Rating: 8.0
   Match Score: 0.081
   Synopsis: A family of undercover superheroes, while trying to live the quiet suburban life, are forced into action to save the world....

3. Vampire Hunter D: Bloodlust (2000)
   Genre: Animation, Action, Fantasy
   IMDB Rating: 7.7
   Match Score: 0.061
   Synopsis: When a girl is abducted by a vampire, a legendary bounty hunter is hired to bring her back....

4. How to Train Your Dragon (2010)
   Genre: Animation, Action, Adventure
   IMDB Rating: 8.1
   Match Score: 0.051
   Synopsis: A hapless young Viking who aspires to hunt dragons becomes the unlikely friend of a young dragon himself, and learns there may be more to the creature...

5. Kôkaku Kidôtai (1995)
   Genre: Animation, Action, Crime
   IMDB Rating: 8.0
   Match Score: 0.05
   Synopsis: A cyborg policewoman and her partner hunt a mysterious and powerful hacker called the Puppet Master....

Describe what kind of movie you're in the mood for (type 'exit' to quit):
> exit
Thank you for using the Enhanced Movie Recommender. Goodbye!

## Results
The system processes the user's input, analyzes it for mood, theme, and time period preferences, and then calculates a similarity score between the input and each movie in the dataset. The top 5 movies with the highest similarity scores are returned as recommendations.






